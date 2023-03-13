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

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Trunk(nn.Module):
  def __init__(self, arch='resnet101', nout=256, no_pretrain=False):
    super(Trunk, self).__init__()
    self.trunk = models.__dict__[arch](pretrained=not no_pretrain)
    ndim = self.trunk.fc.in_features
    self.trunk.fc = Identity() # remove fully-connected layer
    self.fc = nn.Linear(ndim, nout)

  def forward(self, x):
    x = self.trunk(x)
    x = self.fc(x)
    return x


class Classifier(nn.Module):
  def __init__(self, nin=256, nout=13):
    super(Classifier, self).__init__()
    self.fc = nn.Linear(nin, nout)

  def forward(self, x):
    x = self.fc(x)
    return x


class Discriminator(nn.Module):
  def __init__(self, nin=256, nout=2):
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(nin, nin)
    self.fc2 = nn.Linear(nin, nout)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x


def create_model(arch='resnet101', ndim=256, nclasses=13, no_pretrain=False):
  """
      create model from the pretrained model
      remove fully-connected layer and add additional classifier and discriminator
  """

  trunk = Trunk(arch=arch, nout=ndim, no_pretrain=no_pretrain)
  cls = Classifier(nin=ndim, nout=nclasses)
  disc = Discriminator(nin=ndim, nout=2)
  trunk = torch.nn.DataParallel(trunk).cuda()
  cls = torch.nn.DataParallel(cls).cuda()
  disc = torch.nn.DataParallel(disc).cuda()
  return trunk, cls, disc