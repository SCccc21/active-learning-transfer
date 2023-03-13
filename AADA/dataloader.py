import torch, math, os
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import folder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import default_loader
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.datasets.folder import DatasetFolder
import torchvision.datasets as datasets
from torch.utils.data.dataloader import default_collate
# from torch.utils.data.dataloader import _DataLoaderIter
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import mxnet as mx
import lmdb
from tqdm import tqdm
import six
from PIL import Image
import numpy as np

'''
    data augmentation
'''

# IMG_EXTENSIONS += ['.JPG']

class Augmentation(object):
  def __init__(self, dataset=None, augmentation_list=''):
    self.dataset = dataset
    if isinstance(augmentation_list, str):
      augmentation_list = [s for s in augmentation_list.split(',')]
    self.standardize = False
    self.hflip = False
    self.intens_flip = False
    self.intens_scale = False
    self.intens_offset = False
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


class Lighting(object):
  """Lighting noise(AlexNet - style PCA - based noise)"""

  def __init__(self, alphastd=0.1, eigval=None, eigvec=None):
    self.alphastd = alphastd
    if eigval is None:
      self.eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    else:
      self.eigval = eigval
    if eigvec is None:
      self.eigvec = torch.Tensor([[-0.5675,  0.7192,  0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948,  0.4203],])
    else:
      self.eigvec = eigvec

  def __call__(self, img):
    if self.alphastd == 0:
      return img
    alpha = img.new().resize_(3).normal_(0, self.alphastd)
    rgb = self.eigvec.type_as(img).clone()\
        .mul(alpha.view(1, 3).expand(3, 3))\
        .mul(self.eigval.view(1, 3).expand(3, 3))\
        .sum(1).squeeze()
    return img.add(rgb.view(3, 1, 1).expand_as(img))


def data_transformer(dataset=None, augmentation_list=None, train=True, input_range='-1,1'):
  aug = Augmentation(augmentation_list=augmentation_list)
  range_low, range_high = [int(s) for s in input_range.split(',')]
  if dataset == 'MNIST':
    tf = transforms.Compose([
      transforms.Resize([32,32], interpolation=3),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul_(range_high-range_low).add_(range_low)),
      transforms.Lambda(lambda x: aug.augment(x)),
      transforms.Lambda(lambda x: x.repeat(3,1,1)),])
  elif dataset == 'SVHN' or dataset == 'MNIST-M':
    tf = transforms.Compose([
      transforms.Resize([32,32], interpolation=3),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul_(range_high-range_low).add_(range_low)),
      transforms.Lambda(lambda x: aug.augment(x)),])
  return tf


'''
    data loader
'''

def load_small_dbs(root=None, dataset=None, transform=None, augmentation_list=None, input_range='-1,1'):
  if root is None:
    if dataset == 'MNIST':
      root = '/net/acadia6a/data/ksohn/workplace/pytorch/datasets/MNIST/'
    elif dataset == 'SVHN':
      root = '/net/acadia6a/data/ksohn/workplace/pytorch/datasets/SVHN/'
    elif dataset == 'MNIST-M':
      print('not implemented yet')

  if transform is None:
    tf_train = data_transformer(dataset=dataset, augmentation_list=augmentation_list, train=True, input_range=input_range)
    tf_test = data_transformer(dataset=dataset, augmentation_list=augmentation_list, train=False, input_range=input_range)

  if dataset == 'MNIST':
    db_train = datasets.MNIST(root, train=True, transform=tf_train)
    db_test = datasets.MNIST(root, train=False, transform=tf_test)
  elif dataset == 'SVHN':
    db_train = datasets.SVHN(root, split='train', transform=tf_train)
    db_test = datasets.SVHN(root, split='test', transform=tf_test)

  return db_train, db_test


def make_dataset_from_txt_for_merge(root, txt_path, extensions):
  fc = open(txt_path, 'r').readlines()
  images = []
  for i in tqdm(range(len(fc))):
    line = fc[i]
    if len(line.split()) == 1:
      fname = line.split()[0]
      label = 0
    else:
      fname, label = line.split()[:2]
    prefix = fname.split('/')[0]
    if len(prefix.split('__')[1]) == 0:
      r = root
    else:
      r = root[int(prefix.split('__')[1])]
    fname = fname.replace(prefix + '/', '')
    label = int(label)
    if has_file_allowed_extension(fname, extensions):
      path = os.path.join(r, fname)
      if os.path.exists(path):
        item = (path, label)
        images.append(item)
  print(len(images))
  return images


def make_dataset_from_txt(root, txt_path, extensions):
  fc = open(txt_path, 'r').readlines()
  images = []
  for i in tqdm(range(len(fc))):
    line = fc[i]
    if len(line.split()) == 1:
      fname = line.split()[0]
      label = 0
    else:
      fname, label = line.split()[:2]
    label = int(label)
    if has_file_allowed_extension(fname, extensions):
      path = os.path.join(root, fname)
      if os.path.exists(path):
        item = (path, label)
        images.append(item)
  print(len(images))
  return images


def make_dataset_from_txt_remove_duplicate(root, txt_path, extensions):
  fc = open(txt_path, 'r').readlines()
  images = []
  for i in tqdm(range(len(fc))):
    line = fc[i]
    if len(line.split()) == 1:
      fname = line.split()[0]
      label = 0
      cluster = 0
    else:
      fname, label, cluster = line.split()[:3]
    label = int(label)
    cluster = int(cluster)
    if has_file_allowed_extension(fname, extensions):
      path = os.path.join(root, fname)
      if os.path.exists(path):
        item = (path, label, cluster)
        images.append(item)
  print(len(images))
  return images


def make_lmdb_from_txt(root, txt_path, extensions, lmdb_out=None):
  if lmdb_out is None:
    lmdb_out = txt_path.replace('.txt', '_lmdb')
  new_txt_path = lmdb_out + '/data.txt'
  if os.path.exists(os.path.join(lmdb_out, 'data.mdb')):
    return lmdb_out, new_txt_path
  images = make_dataset_from_txt(root, txt_path, extensions)
  
  env = lmdb.open(lmdb_out, map_size=1000*2**30)
  with env.begin(write=True) as txn:
    for i in tqdm(range(len(images))):
      path = images[i][0]
      with open(path, 'rb') as fd:
        rawBytes = fd.read()
        txn.put(path.replace(root, '').encode(), rawBytes)
  g = open(new_txt_path, 'w')
  for i in range(len(images)):
    path = images[i][0]
    label = images[i][1]
    g.write(path.replace(root, '') + ' %d\n' %label)
  g.close()
  return lmdb_out, new_txt_path


def make_lmdb_from_txt_remove_duplicate(root, txt_path, extensions, lmdb_out=None):
  if lmdb_out is None:
    lmdb_out = txt_path.replace('.txt', '_lmdb')
  new_txt_path = lmdb_out + '/data.txt'
  if os.path.exists(os.path.join(lmdb_out, 'data.mdb')):
    return lmdb_out, new_txt_path
  images = make_dataset_from_txt_remove_duplicate(root, txt_path, extensions)
  env = lmdb.open(lmdb_out, map_size=1000*2**30)
  with env.begin(write=True) as txn:
    for i in tqdm(range(len(images))):
      path = images[i][0]
      with open(path, 'rb') as fd:
        rawBytes = fd.read()
        txn.put(path.replace(root, '').encode(), rawBytes)
  g = open(new_txt_path, 'w')
  for i in range(len(images)):
    path = images[i][0]
    label = images[i][1]
    cluster = images[i][2]
    g.write(path.replace(root, '') + ' %d %d\n' %(label, cluster))
  g.close()
  return lmdb_out, new_txt_path


class NPairSamplerRecordIO(Sampler):
  def __init__(self, dataset, batch_size, n_batch=100, n_min=25):
    self.record = dataset.record
    # make dictionary
    dct = {}
    k = 0
    for key in self.record.keys:
      k += 1
      if k % 100000 == 0:
        print('%07d/%07d' %(k, len(self.record.keys)))
      s = self.record.read_idx(key)
      header, _ = mx.recordio.unpack(s)
      if header.flag > 0:
        continue
      label = int(header.label)
      if label not in dct:
        dct[label] = []
      dct[label].append(key)
    # remove if class has less than n_min images
    for label in sorted(dct.keys()):
      if len(dct[label]) < n_min:
        del dct[label]
    # relabel from 0
    self.nClasses = len(dct)
    dct_new = {}
    new_label = 0
    for label in sorted(dct.keys()):
      dct_new[new_label] = dct[label]
      new_label += 1
    self.dct = dct_new
    self.batch_size = batch_size
    self.n_batch = n_batch

  def __iter__(self):
    indices = []
    shuffled_classes = torch.randperm(self.nClasses).tolist()
    for i in range(self.n_batch):
      # 1. shuffle
      if len(shuffled_classes) < self.batch_size:
        shuffled_classes = torch.randperm(self.nClasses).tolist()
      # 2. subsample
      indices1 = []
      indices2 = []
      for j in range(self.batch_size):
        class_idx = shuffled_classes.pop()
        sample_idx = torch.randperm(len(self.dct[class_idx]))
        indices1.append(self.dct[class_idx][sample_idx[0]])
        indices2.append(self.dct[class_idx][sample_idx[1]])
      indices += indices1 + indices2
    return iter(indices)
  
  def __len__(self):
    return self.n_batch*self.batch_size*2


class AFSampler(Sampler):
  def __init__(self, dataset, batch_size, n_batch=100):
    # self.samples = [(line.split()[0].encode(), int(line.split()[1]), int(line.split()[2]), int(line.split()[3])) for line in open(txt_path, 'r').readlines()]
    self.samples = dataset.samples
    # make dictionary
    dct = {}
    for i in range(len(self.samples)):
      label = '%04d_%04d' %(self.samples[i][1], self.samples[i][2])
      if label not in dct:
        dct[label] = {}
      if self.samples[i][3] not in dct[label]:
        dct[label][self.samples[i][3]] = []
      dct[label][self.samples[i][3]].append(i)
    self.dct = dct
    self.batch_size = batch_size
    self.n_batch = n_batch
    self.class_list = []
    for label in sorted(dct.keys()):
      self.class_list.append(label)
    self.nClasses = len(self.class_list)
  
  def __iter__(self):
    indices = []
    shuffled_classes = torch.randperm(self.nClasses).tolist()
    for i in range(self.n_batch):
      # 1. shuffle
      if len(shuffled_classes) < self.batch_size:
        shuffled_classes = torch.randperm(self.nClasses).tolist()
      # 2. subsample
      indices1 = []
      indices2 = []
      for j in range(self.batch_size):
        class_idx = shuffled_classes.pop()
        class_idx = self.class_list[class_idx]
        sample_idx1 = torch.randperm(len(self.dct[class_idx][0]))
        k = int(torch.randperm(len(self.dct[class_idx]))[0])
        sample_idx2 = torch.randperm(len(self.dct[class_idx][k]))
        indices1.append(self.dct[class_idx][0][sample_idx1[0]])
        indices2.append(self.dct[class_idx][k][sample_idx2[0]])
      indices += indices1 + indices2
    return iter(indices)
  
  def __len__(self):
    return self.n_batch*self.batch_size*2


class AFSampler2(Sampler):
  def __init__(self, dataset, batch_size, n_batch=100, separation=1):
    # self.samples = [(line.split()[0].encode(), int(line.split()[1]), int(line.split()[2]), int(line.split()[3])) for line in open(txt_path, 'r').readlines()]
    self.samples = dataset.samples
    # make dictionary
    dct = {}
    for i in range(len(self.samples)):
      label = '%04d_%04d' %(self.samples[i][1], self.samples[i][2])
      if label not in dct:
        dct[label] = {}
      if self.samples[i][3] not in dct[label]:
        dct[label][self.samples[i][3]] = []
      dct[label][self.samples[i][3]].append(i)
    self.dct = dct
    self.batch_size = batch_size
    self.n_batch = n_batch
    self.class_list = []
    self.separation = separation
    for label in sorted(dct.keys()):
      self.class_list.append(label)
    self.nClasses = len(self.class_list)
  
  def __iter__(self):
    indices = []
    shuffled_classes = torch.randperm(self.nClasses).tolist()
    for i in range(self.n_batch):
      # 1. shuffle
      if len(shuffled_classes) < self.batch_size:
        shuffled_classes = torch.randperm(self.nClasses).tolist()
      # 2. subsample
      indices1 = []
      indices2 = []
      for j in range(self.batch_size):
        class_idx = shuffled_classes.pop()
        class_idx = self.class_list[class_idx]
        #sample_idx = torch.randperm(len(self.dct[class_idx])-3).add(3).numpy()
        indices1.append(self.dct[class_idx][len(self.dct[class_idx])-1][0])
        #indices2.append(self.dct[class_idx][len(self.dct[class_idx])-1-sample_idx[0]][0])
        indices2.append(self.dct[class_idx][max(len(self.dct[class_idx])-1-self.separation, 0)][0])
      indices += indices1 + indices2
    return iter(indices)
  
  def __len__(self):
    return self.n_batch*self.batch_size*2


class NPairSampler(Sampler):
  def __init__(self, dataset, batch_size, n_batch=100, min_size=20):
    self.samples = dataset.samples
    # make dictionary
    dct = {}
    for i in range(len(self.samples)):
      label = self.samples[i][1]
      if label not in dct:
        dct[label] = []
      dct[label].append(i)
    self.dct = dct
    self.batch_size = batch_size
    self.n_batch = n_batch
    self.class_list = []
    for label in sorted(dct.keys()):
      if len(dct[label]) >= min_size:
        self.class_list.append(label)
    self.nClasses = len(self.class_list)
  
  def __iter__(self):
    indices = []
    shuffled_classes = torch.randperm(self.nClasses).tolist()
    for i in range(self.n_batch):
      # 1. shuffle
      if len(shuffled_classes) < self.batch_size:
        shuffled_classes = torch.randperm(self.nClasses).tolist()
      # 2. subsample
      indices1 = []
      indices2 = []
      for j in range(self.batch_size):
        class_idx = shuffled_classes.pop()
        class_idx = self.class_list[class_idx]
        sample_idx = torch.randperm(len(self.dct[class_idx]))
        indices1.append(self.dct[class_idx][sample_idx[0]])
        indices2.append(self.dct[class_idx][sample_idx[1]])
      indices += indices1 + indices2
    return iter(indices)
  
  def __len__(self):
    return self.n_batch*self.batch_size*2


class NPairSampler_ND(Sampler):
  """ remove near-duplicate """
  def __init__(self, dataset, batch_size, n_batch=100, min_size=10):
    self.samples = dataset.samples
    # make dictionary
    dct = {}
    for i in range(len(self.samples)):
      label = self.samples[i][1]
      cluster = self.samples[i][2]
      if label not in dct:
        dct[label] = {}
      if cluster not in dct[label]:
        dct[label][cluster] = []
      dct[label][cluster].append(i)
    self.dct = dct
    self.batch_size = batch_size
    self.n_batch = n_batch
    self.class_list = []
    for label in sorted(dct.keys()):
      if len(dct[label]) >= min_size:
        self.class_list.append(label)
    self.nClasses = len(self.class_list)
  
  def __iter__(self):
    indices = []
    shuffled_classes = torch.randperm(self.nClasses).tolist()
    for i in range(self.n_batch):
      # 1. shuffle
      if len(shuffled_classes) < self.batch_size:
        shuffled_classes = torch.randperm(self.nClasses).tolist()
      # 2. subsample
      indices1 = []
      indices2 = []
      for j in range(self.batch_size):
        class_idx = shuffled_classes.pop()
        class_idx = self.class_list[class_idx]
        sample_idx = torch.randperm(len(self.dct[class_idx])).numpy()
        sample_idx1 = torch.randperm(len(self.dct[class_idx][sample_idx[0]])).numpy()
        sample_idx2 = torch.randperm(len(self.dct[class_idx][sample_idx[1]])).numpy()
        indices1.append(self.dct[class_idx][sample_idx[0]][sample_idx1[0]])
        indices2.append(self.dct[class_idx][sample_idx[1]][sample_idx2[0]])
      indices += indices1 + indices2
    return iter(indices)
  
  def __len__(self):
    return self.n_batch*self.batch_size*2


class NPairRandomSampler(Sampler):
  def __init__(self, dataset, batch_size, n_batch=100):
    self.samples = dataset.samples
    self.list_of_target = []
    # make dictionary
    dct = {}
    for i in range(len(self.samples)):
      label = self.samples[i][1]
      if label not in dct:
        dct[label] = {}
        dct[label]['idx'] = []
        dct[label]['sample_idx'] = []
      dct[label]['idx'].append(i)
      self.list_of_target.append(label)
    self.dct = dct
    self.batch_size = batch_size
    self.n_batch = n_batch
    self.nClasses = len(dct)

  def __iter__(self):
    indices = []
    shuffled_indices = torch.randperm(len(self.list_of_target)).tolist()
    for i in range(self.n_batch):
      # 1. shuffle
      if len(shuffled_indices) < self.batch_size:
        shuffled_indices = torch.randperm(len(self.list_of_target)).tolist()
      # 2. subsample
      indices1 = []
      indices2 = []
      for j in range(self.batch_size):
        class_idx = self.list_of_target[shuffled_indices.pop()]
        if len(self.dct[class_idx]['sample_idx']) < 2:
          self.dct[class_idx]['sample_idx'] = torch.randperm(len(self.dct[class_idx]['idx'])).tolist()
        indices1.append(self.dct[class_idx]['idx'][self.dct[class_idx]['sample_idx'].pop()])
        indices2.append(self.dct[class_idx]['idx'][self.dct[class_idx]['sample_idx'].pop()])
      indices += indices1 + indices2
    return iter(indices)
  
  def __len__(self):
    return self.n_batch*self.batch_size*2


class BatchSampler(Sampler):
  def __init__(self, sampler, batch_size):
    self.sampler = sampler
    self.batch_size = 2*batch_size
  
  def __iter__(self):
    batch = []
    for idx in self.sampler:
      batch.append(idx)
      if len(batch) == self.batch_size:
        yield batch
        batch = []
  
  def __len__(self):
    return self.sampler.__len__() // self.batch_size


class ImageFromTextForMerge(DatasetFolder):
  def __init__(self, root, txt_path, loader=default_loader, transform=None, target_transform=None, extensions=IMG_EXTENSIONS):
    samples = make_dataset_from_txt_for_merge(root, txt_path, extensions)
    if len(samples) == 0:
      raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                         "Supported extensions are: " + ",".join(extensions)))
    self.root = root
    self.txt_path = txt_path
    self.loader = loader
    self.samples = samples
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return sample.mul_(255.0), target


class ImageFromText(DatasetFolder):
  def __init__(self, root, txt_path, loader=default_loader, transform=None, target_transform=None, extensions=IMG_EXTENSIONS):
    samples = make_dataset_from_txt(root, txt_path, extensions)
    if len(samples) == 0:
      raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                         "Supported extensions are: " + ",".join(extensions)))
    self.root = root
    self.txt_path = txt_path
    self.loader = loader
    self.samples = samples
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return sample.mul_(255.0), target


class ImageMaskFromLMDB(DatasetFolder):
  def __init__(self, root, txt_path, txt_path_mask, lmdb_out=None, lmdb_mask=None, transform=None, transform_mask=None, target_transform=None, extensions=IMG_EXTENSIONS):
    if lmdb_out is None:
      lmdb_out = txt_path.replace('.txt', '_lmdb')
    if lmdb_mask is None:
      lmdb_mask = lmdb_out + '_mask'
    lmdb_out, _ = make_lmdb_from_txt(root, txt_path, extensions, lmdb_out)
    lmdb_mask, _ = make_lmdb_from_txt(root, txt_path_mask, extensions, lmdb_mask)
    self.env = lmdb.open(lmdb_out, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    self.env_mask = lmdb.open(lmdb_mask, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    self.root = root
    # 4-tuple, image path, id, view1 (yaw), view2 (elevation)
    self.samples = [(line.split()[0].encode(), int(line.split()[1]), int(line.split()[2]), int(line.split()[3])) for line in open(txt_path, 'r').readlines()]
    self.samples_mask = [(line.split()[0].encode(), int(line.split()[1]), int(line.split()[2]), int(line.split()[3])) for line in open(txt_path_mask, 'r').readlines()]
    self.transform = transform
    self.transform_mask = transform_mask
    self.target_transform = target_transform

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    # image
    env = self.env
    key, _, _, target = self.samples[index]
    with env.begin(write=False) as txn:
      imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    sample = Image.open(buf).convert('RGB')
    if self.transform is not None:
      sample = self.transform(sample)
    # mask
    env_mask = self.env_mask
    key, _, _, target = self.samples_mask[index]
    with env_mask.begin(write=False) as txn:
      imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    sample_mask = Image.open(buf).convert('RGB')
    if self.transform_mask is not None:
      sample_mask = self.transform_mask(sample_mask)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return sample.mul_(2.0).add_(-1.0), sample_mask, target


class ImageFromLMDBforAF(DatasetFolder):
  def __init__(self, root, txt_path, lmdb_out=None, transform=None, target_transform=None, extensions=IMG_EXTENSIONS):
    if lmdb_out is None:
      lmdb_out = txt_path.replace('.txt', '_lmdb')
    lmdb_out, _ = make_lmdb_from_txt(root, txt_path, extensions, lmdb_out)
    self.env = lmdb.open(lmdb_out, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    self.root = root
    # 4-tuple, image path, id, view1 (yaw), view2 (elevation)
    self.samples = [(line.split()[0].encode(), int(line.split()[1]), int(line.split()[2]), int(line.split()[3])) for line in open(txt_path, 'r').readlines()]
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    # image
    env = self.env
    key, _, _, target = self.samples[index]
    with env.begin(write=False) as txn:
      imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    sample = Image.open(buf).convert('RGB')
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return sample.mul_(2.0).add_(-1.0), target


class ImageFromLMDB(DatasetFolder):
  def __init__(self, root, txt_path, lmdb_out=None, transform=None, target_transform=None, extensions=IMG_EXTENSIONS):
    if lmdb_out is None:
      lmdb_out = txt_path.replace('.txt', '_lmdb')
    lmdb_out, txt_path = make_lmdb_from_txt(root, txt_path, extensions, lmdb_out)
    self.env = lmdb.open(lmdb_out, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with self.env.begin(write=False) as txn:
      self.length = txn.stat()['entries']
    self.root = root
    self.samples = [(line.split()[0].encode(), int(line.split()[1])) for line in open(txt_path, 'r').readlines()]
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    env = self.env
    key, target = self.samples[index]
    with env.begin(write=False) as txn:
      imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    sample = Image.open(buf).convert('RGB')
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return sample.mul_(255.0), target
    #return sample.mul_(2.0).add_(-1.0), target


class ImageFromLMDB_ND(DatasetFolder):
  """remove near duplicate"""
  def __init__(self, root, txt_path, lmdb_out=None, transform=None, target_transform=None, extensions=IMG_EXTENSIONS):
    if lmdb_out is None:
      lmdb_out = txt_path.replace('.txt', '_lmdb')
    lmdb_out, txt_path = make_lmdb_from_txt_remove_duplicate(root, txt_path, extensions, lmdb_out)
    self.env = lmdb.open(lmdb_out, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with self.env.begin(write=False) as txn:
      self.length = txn.stat()['entries']
    self.root = root
    self.samples = [(line.split()[0].encode(), int(line.split()[1]), int(line.split()[2])) for line in open(txt_path, 'r').readlines()]
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    env = self.env
    key, target = self.samples[index][:2]
    with env.begin(write=False) as txn:
      imgbuf = txn.get(key)
    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    sample = Image.open(buf).convert('RGB')
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return sample.mul_(255.0), target
    #return sample.mul_(2.0).add_(-1.0), target


class ImageFromRecordIO(DatasetFolder):
  def __init__(self, root, idx_file, rec_file, transform=None, target_transform=None):    
    self.root = root
    self.samples = [(1,2)]
    self.record = mx.recordio.MXIndexedRecordIO(os.path.join(self.root, idx_file), os.path.join(self.root, rec_file), 'r')
    self.transform = transform
    self.target_transform = target_transform

  def _loader(self, key):
    s = self.record.read_idx(key)
    header, sample = mx.recordio.unpack_img(s)
    target = int(header.label)
    return sample, target

  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    sample, target = self._loader(index)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)
    return sample.mul_(255.0), target

