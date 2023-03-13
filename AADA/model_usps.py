import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import pdb

class Augmentation(object):
  def __init__(self, dataset=None, augmentation_list=''):
    self.dataset = dataset
    if isinstance(augmentation_list, str):
      augmentation_list = [s for s in augmentation_list.split(',')]
    self.pixel_flip = False
    self.standardize = False
    self.hflip = False
    self.intens_flip = False
    self.intens_scale = False
    self.intens_offset = False
    if 'pixel_flip' in augmentation_list or 'pf' in augmentation_list:
      self.pixel_flip = True
    if 'standardize' in augmentation_list or 's' in augmentation_list:
      self.standardize = True
    if 'hflip' in augmentation_list or 'hf' in augmentation_list:
      self.hflip = True
    if 'intens_flip' in augmentation_list or 'if' in augmentation_list:
      self.intens_flip = True
    if 'intens_scale' in augmentation_list or 'is' in augmentation_list:
      self.intens_scale = True
      self.intens_scale_range_lower = 0.25
      self.intens_scale_range_upper = 1.5
    if 'intens_offset' in augmentation_list or 'io' in augmentation_list:
      self.intens_offset = True
      self.intens_offset_range_lower = -0.5
      self.intens_offset_range_upper = 0.5

  def augment(self, X):
    X = X.numpy()
    if self.pixel_flip:
      col_factor = (np.random.binomial(1, 0.5, size=(1, 1, 1)) * 2 - 1).astype(np.float32)
      X = (X * col_factor).astype(np.float32)
    if self.standardize:
      # assuming numpy array
      X = X - X.mean(axis=(0,1,2), keepdims=True)
      X = X / X.std(axis=(0,1,2), keepdims=True)
    if self.intens_flip:
      col_factor = (np.random.binomial(1, 0.5, size=(1, 1, 1)) * 2 - 1).astype(np.float32)
      X = (X * col_factor).astype(np.float32)
    if self.intens_scale:
      col_factor = np.random.uniform(low=self.intens_scale_range_lower, high=self.intens_scale_range_upper, size=(1, 1, 1))
      X = (X * col_factor).astype(np.float32)
    if self.intens_offset:
      col_offset = np.random.uniform(low=self.intens_offset_range_lower, high=self.intens_offset_range_upper, size=(1, 1, 1))
      X = (X + col_offset).astype(np.float32)
    X = torch.Tensor(X)
    return X


def data_transformer(dataset=None, augmentation_list=None, train=True, input_range='-1,1'):
  aug = Augmentation(augmentation_list=augmentation_list)
  range_low, range_high = [int(s) for s in input_range.split(',')]
  if dataset.upper() == 'MNIST':
    tf = transforms.Compose([
      # transforms.Resize([32,32], interpolation=3), #NOTE
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul_(range_high-range_low).add_(range_low)),
      transforms.Lambda(lambda x: aug.augment(x)),
      # transforms.Lambda(lambda x: x.repeat(3,1,1)),
      ])
  elif dataset.upper() == 'SVHN' or dataset.upper() == 'MNIST-M':
    tf = transforms.Compose([
      transforms.Resize([32,32], interpolation=3),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul_(range_high-range_low).add_(range_low)),
      transforms.Lambda(lambda x: aug.augment(x))
      ])
  elif dataset.upper() == 'USPS':
    tf = transforms.Compose([
      transforms.Resize([28,28], interpolation=3),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul_(range_high-range_low).add_(range_low)),
      transforms.Lambda(lambda x: aug.augment(x))
      ])
  return tf


def load_dbs(root=None, dataset=None, transform=None, augmentation_list=None, input_range='-1,1'):
  if root is None:
    if dataset.upper() == 'MNIST':
      root = '/home/chensi/data/MNIST/'
    elif dataset.upper() == 'SVHN':
      root = '/home/chensi/data/svhn/'
    elif dataset.upper() == 'USPS':
      root = '/home/chensi/data/usps/'
    elif dataset.upper() == 'MNIST-M':
      print('not implemented yet')

  if transform is None:
    tf_train = data_transformer(dataset=dataset, augmentation_list=augmentation_list, train=True, input_range=input_range)
    tf_test = data_transformer(dataset=dataset, augmentation_list=augmentation_list.replace('if','').replace('is', '').replace('io', ''), train=False, input_range=input_range)

  if dataset.upper() == 'MNIST':
    db_train = datasets.MNIST(root, train=True, transform=tf_train, target_transform=int, download=True)
    db_test = datasets.MNIST(root, train=False, transform=tf_test, target_transform=int, download=True)
    db_train_noaug = datasets.MNIST(root, train=True, transform=tf_test, target_transform=int, download=True)
  elif dataset.upper() == 'SVHN':
    db_train = datasets.SVHN(root, split='train', transform=tf_train, target_transform=int, download=True)
    db_test = datasets.SVHN(root, split='test', transform=tf_test, target_transform=int, download=True)
    db_train_noaug = datasets.SVHN(root, split='train', transform=tf_test, target_transform=int, download=True)

  elif dataset.upper() == 'USPS':
    db_train = datasets.USPS(root, train=True, transform=tf_train, target_transform=int, download=True)
    db_test = datasets.USPS(root, train=False, transform=tf_test, target_transform=int, download=True)
    db_train_noaug = datasets.USPS(root, train=True, transform=tf_test, target_transform=int, download=True)

  return db_train, db_test, db_train_noaug


class Trunk(nn.Module):
  def __init__(self, arch='convnet', nout=500):
    super(Trunk, self).__init__()
 
    if arch == 'convnet':
      self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
      self.relu = nn.ReLU(inplace=True)
      self.dropout = nn.Dropout(p=0.5)
      self.conv1 = nn.Conv2d(1,20,kernel_size=5,stride=1,padding=1,bias=True)
      self.conv2 = nn.Conv2d(20,50,kernel_size=5,stride=1,padding=1,bias=True)
      self.out_view = 1250
      self.nout = nout
      self.fc = nn.Linear(self.out_view,nout)
    else:
      print('not implemented')
      sys.exit(1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.dropout(x)
    x = self.maxpool(x)
    x = self.relu(x)

    # pdb.set_trace()
    x = x.view(x.shape[0],-1)
    x = self.fc(x)
    x = self.relu(x)
    x = self.dropout(x)

    return x


class Classifier(nn.Module):
  def __init__(self, nin=500, nout=10):
    super(Classifier, self).__init__()
    self.fc = nn.Linear(nin, nout)

  def forward(self, x):
    x = self.fc(x)
    return x


class Discriminator(nn.Module):
  def __init__(self, nin=500, nout=2):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(nin, nin)
    self.fc2 = nn.Linear(nin, nin)
    self.fc3 = nn.Linear(nin, nout)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x


def create_model(arch='convnet', ndim=500, nclasses=10, no_pretrain=False):
  """
      create model from scratch
      remove fully-connected layer and add additional classifier and discriminator
  """

  trunk = Trunk(arch=arch, nout=ndim)
  cls = Classifier(nin=ndim, nout=nclasses)
  disc = Discriminator(nin=ndim, nout=2)
  trunk = torch.nn.DataParallel(trunk).cuda()
  cls = torch.nn.DataParallel(cls).cuda()
  disc = torch.nn.DataParallel(disc).cuda()
  return trunk, cls, disc
