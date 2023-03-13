import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import keras
from keras.layers import Input, Flatten, Dense, LeakyReLU, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K
from keras.models import clone_model
from keras import Model
from keras.datasets import mnist
from keras.utils import np_utils

import tensorflow.compat.v1 as tf
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

import numpy as np
import collections
import copy
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.optimize as op 
import seaborn as sns
import pickle


import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import math
from cycada import ResNetClassifier


def torch_visda_resnet_data_to_acc(x_train, y_train, x_test, y_test, verbose=0):

  criterion = torch.nn.CrossEntropyLoss()
  net = ResNetClassifier(num_cls=12).cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=16, shuffle=True, drop_last=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, drop_last=True)

  for epoch in range(20):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = 100 * correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))
  return accuracy.item()



def torch_visda_resnet_data_to_acc_multiple(X_train, y_train, x_val, y_val, repeat=3, verbose=0):
  acc_lst = []
  for _ in range(repeat):
    acc = torch_visda_resnet_data_to_acc(X_train, y_train, x_val, y_val, verbose)
    acc_lst.append(acc)
  return np.mean(acc_lst)


def torch_visda_resnet_data_to_net(x_train, y_train, x_test, y_test, n_epoch=10, verbose=0):

  criterion = torch.nn.CrossEntropyLoss()
  net = ResNetClassifier(num_cls=12).cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=64, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = 100 * correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))
  return net





class ResBase101(nn.Module):
    def __init__(self):
        super(ResBase101, self).__init__()
        model_resnet101 = torchvision.models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResClassifier(nn.Module):
    def __init__(self, class_num=12, extract=False):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.fc3 = nn.Linear(1000, class_num)
        self.extract = extract

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(0.5))
        fc2_emb = self.fc2(fc1_emb)
        if self.training:
            fc2_emb.mul_(math.sqrt(0.5))            
        logit = self.fc3(fc2_emb)

        if self.extract:
            return fc2_emb, logit
        return logit

class ResBase18(nn.Module):
    def __init__(self):
        super(ResBase18, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3 # 1024
        # self.layer4 = model_resnet18.layer4
        # self.avgpool = model_resnet18.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

'''
class ResClassifier(nn.Module):
    def __init__(self, class_num=12, extract=False):
        super(ResClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout()
            )
        self.fc3 = nn.Linear(512, class_num)
        self.extract = extract

    def forward(self, x):
        fc1_emb = self.fc1(x)
        if self.training:
            fc1_emb.mul_(math.sqrt(0.5))
        # fc2_emb = self.fc2(fc1_emb)
        # if self.training:
        #     fc2_emb.mul_(math.sqrt(0.5))            
        logit = self.fc3(fc1_emb)

        if self.extract:
            return fc1_emb, logit
        return logit
'''


# def featureExtract(x, netG, netF):
#     with torch.no_grad():
#         s_bottleneck = netG(x.cuda())
#         emb, s_logit = netF(s_bottleneck)
#     return emb.view(emb.size(0), -1).cpu().detach()



def featureExtract(x, netG, netF):
    batch_size = 64
    train_size = x.size(0)
    x_feat = torch.zeros(train_size, 1000)
    x = x.cuda()

    with torch.no_grad():
        for j in range(train_size//batch_size):
            start_ind = j*batch_size
            s_bottleneck = netG(x[start_ind: min(start_ind+batch_size, train_size)])
            x_f, _ = netF(s_bottleneck)
            x_f = x_f.view(x_f.size(0), -1)
            x_feat[start_ind: min(start_ind+batch_size, train_size)] = x_f

    return x_feat.detach().cpu().numpy()





from torch.utils.data import Dataset, DataLoader, TensorDataset
class torchLogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(torchLogisticRegression, self).__init__()
        self.flatten = Flatten()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.input_dim = input_dim

    def forward(self, x, last=False):
        outputs = self.linear(x)
        if last:
            return outputs, x
        else:
            return outputs

    def get_embedding_dim(self):
        return self.input_dim



def torch_encoder_logistic_data_to_net(x_train, y_train, x_test, y_test, verbose=0):

  criterion = torch.nn.CrossEntropyLoss()
  net = torchLogisticRegression(x_train.shape[1], 12)
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train), torch.Tensor(y_train)
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test), torch.Tensor(y_test)
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(5):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels.long())
          # labels = torch.argmax(labels, dim=1)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          labels = labels.long()
          images = Variable(images)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          # labels = torch.argmax(labels, dim=1)
          correct += (predicted == labels).sum()
      accuracy = 100 * correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))
  print('Model Accuracy: {}'.format(accuracy))
  return net


# cluster_syn = pickle.load(open(sample_dir+'/cluster_syn_feat.data', 'rb'))

def setTrainFalse(layer):
  for i in range(4):
    layer[i].conv1.weight.trainable = False
    layer[i].conv2.weight.trainable = False



from scipy.stats import wasserstein_distance
def get_label(centroids, feat):
  best_dist = 1e9
  for k in range(len(centroids)):
    # import pdb;pdb.set_trace()
    dist = wasserstein_distance(centroids[k].detach().cpu(), feat.detach().cpu())
    if dist < best_dist:
      best_dist = dist 
      pred_label = k 
  
  return pred_label

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def data_to_acc_finetune(adda_net_file, x_train, y_train, x_test, y_test, x_unlabel=None, verbose=0, lr=2e-6):
  netG = ResBase101().cuda()
  netF = ResClassifier(class_num=12, extract=True).cuda()
  netF.train()
  freeze(netG)
  netG.load_state_dict(torch.load(adda_net_file[0]))
  netF.load_state_dict(torch.load(adda_net_file[1]))


  # net.conv1.weight.trainable = False
  # setTrainFalse(net.layer1)
  # setTrainFalse(net.layer2)
  # setTrainFalse(net.layer3)
  

  optimizer = optim.Adam(netF.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)
  criterion = nn.CrossEntropyLoss()

  if x_unlabel is not None:
  # get centriods for K classes
    masks = []
    centroids = []
    for k in range(12):
      mask = np.zeros_like(y_train, dtype=bool)
      for i in range(len(y_train)):
        mask[i] = True if y_train[i] == k else False 
    
      cluster_x = x_train[mask]
      if cluster_x.shape[0] > 1:
        centroid = featureExtract(cluster_x, netG, netF)
        centroid = torch.mean(centroid, dim=0)
      else:
        print('Unnormal Alert!')
        centroid = cluster_syn[k]

      masks.append(mask)
      centroids.append(centroid)

    # predict fake labels for unlabel data
    x_unlabel_feat = []
    for j in range(x_unlabel.shape[0]//50):
      start_ind = j*50
      batch_X, batch_y = [], []
      stop_ind = min(start_ind+50, x_unlabel.shape[0])
      feat = featureExtract(x_unlabel[start_ind:stop_ind], netG, netF)
      x_unlabel_feat.append(feat)

    x_unlabel_feat = torch.cat(x_unlabel_feat, dim=0)

    y_unlabel = []
    for i in range(x_unlabel.size(0)):
      y_unlabel.append(get_label(centroids, x_unlabel_feat[i]))

    y_unlabel = torch.tensor(y_unlabel)

    x_train = torch.cat((x_train, x_unlabel), dim=0)
    y_train = torch.cat((y_train, y_unlabel), dim=0)

  finetune_dataset = TensorDataset(x_train,y_train)
  train_loader = DataLoader(dataset=finetune_dataset, batch_size=32, shuffle=True)
  test_dataset = TensorDataset(x_test,y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  best_acc = 0
  for epoch in range(35):
      netF.train()
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images.cuda(), requires_grad=False)
          labels = Variable(labels.cuda(), requires_grad=False)
          optimizer.zero_grad()
          # import pdb;pdb.set_trace()
          s_bottleneck = netG(images)
          _, score = netF(s_bottleneck)
          loss = criterion(score, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      test_acc = get_da_acc(netG, netF, x_test, y_test)

      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), test_acc*100.0))

      if test_acc > best_acc:
        best_acc = test_acc
      else:
        break
        
  print('Model Accuracy: {}'.format(best_acc*100.0))
  return best_acc


def get_unlabel(selected_rank):
  idx_unlabel = np.ones_like(visda_y_unlabel_re, dtype=bool)
  idx_unlabel[selected_rank] = False
  return visda_x_unlabel_re[idx_unlabel]

def get_da_acc(netG, netF, x_test, y_test):
  
  netG.eval()
  netF.eval()
  with torch.no_grad():
    test_dataset = TensorDataset(x_test,y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.cuda(), requires_grad=False)
        labels = Variable(labels.cuda(), requires_grad=False)
        s_bottleneck = netG(images)
        _, score = netF(s_bottleneck)
        pred = score.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(labels.data).cpu().sum()
        total += labels.size(0)
      
  accuracy = 100.0 * correct/total

  return accuracy / 100