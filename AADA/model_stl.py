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
  if dataset.upper() == 'CIFAR':
    tf = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: aug.augment(x)),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))
      ])
  elif dataset.upper() == 'STL':
    tf = transforms.Compose([
      transforms.Resize([32,32]),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: aug.augment(x)),
      transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
      ])
  return tf


def load_dbs(root=None, dataset=None, transform=None, augmentation_list=None, input_range='-1,1'):
  if root is None:
    if dataset.upper() == 'CIFAR':
      root = '/home/chensi/data/cifar10/'
    elif dataset.upper() == 'STL':
      root = '/home/chensi/data/stl10/'


  if transform is None:
    tf_train = data_transformer(dataset=dataset, augmentation_list=augmentation_list, train=True, input_range=input_range)
    tf_test = data_transformer(dataset=dataset, augmentation_list=augmentation_list.replace('if','').replace('is', '').replace('io', ''), train=False, input_range=input_range)

  if dataset.upper() == 'CIFAR':
    db_train = datasets.CIFAR10(root, train=True, transform=tf_train, target_transform=int, download=True)
    db_test = datasets.CIFAR10(root, train=False, transform=tf_test, target_transform=int, download=True)
    db_train_noaug = datasets.CIFAR10(root, train=True, transform=tf_test, target_transform=int, download=True)

  elif dataset.upper() == 'STL':
    db_train = datasets.STL10(root, split='train', transform=tf_train, target_transform=int, download=True)
    db_test = datasets.STL10(root, split='test', transform=tf_test, target_transform=int, download=True)
    db_train_noaug = datasets.STL10(root, split='train', transform=tf_test, target_transform=int, download=True)

  return db_train, db_test, db_train_noaug


class Trunk(nn.Module):
  def __init__(self, arch='convnet', nout=500):
    super(Trunk, self).__init__()
 
    if arch == 'convnet':
      self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
      self.relu = nn.ReLU(inplace=True)
      self.dropout = nn.Dropout(p=0.5)
      self.conv1 = nn.Conv2d(3,20,kernel_size=5,stride=1,padding=1,bias=True)
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


# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        residual = x 
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual

class Downsample(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x):
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

class ResNetCifar(nn.Module):
    def __init__(self, depth, width=1, block=BasicBlock, classes=10, channels=3):
        assert (depth - 2) % 6 == 0
        self.N = (depth - 2) // 6
        super(ResNetCifar, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(block, 16 * width)
        self.layer2 = self._make_layer(block, 32 * width, stride=2)
        self.layer3 = self._make_layer(block, 64 * width, stride=2)
        self.bn = nn.BatchNorm2d(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(64 * width, classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(self.inplanes, planes, stride)
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes
        for i in range(self.N - 1):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)  #feature
        # pdb.set_trace()
        x = self.avgpool(x)
        # pdb.set_trace()
        x = x.view(x.size(0), -1)
        return x
        

class Classifier(nn.Module):
  def __init__(self, nin=512, nout=10):
    super(Classifier, self).__init__()
    self.fc = nn.Linear(nin, nout)

  def forward(self, x):
    x = self.fc(x)
    return x


class Discriminator(nn.Module):
  def __init__(self, nin=512, nout=2):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(nin, nin)
    self.fc2 = nn.Linear(nin, nin)
    self.fc3 = nn.Linear(nin, nout)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    # pdb.set_trace()
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x


def create_model(arch='convnet', ndim=512, nclasses=10, no_pretrain=False):
  """
      create model from scratch
      remove fully-connected layer and add additional classifier and discriminator
  """

#   trunk = Trunk(arch=arch, nout=ndim)
  trunk = ResNetCifar(26, 2, classes=10, channels=3)
  cls = Classifier(nin=ndim, nout=nclasses)
  disc = Discriminator(nin=ndim, nout=2)
  trunk = torch.nn.DataParallel(trunk).cuda()
  cls = torch.nn.DataParallel(cls).cuda()
  disc = torch.nn.DataParallel(disc).cuda()
  return trunk, cls, disc
