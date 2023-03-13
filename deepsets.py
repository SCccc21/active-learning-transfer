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


"""### Sample Utility"""

# utility_func_args = [x_train, y_train, x_val, y_val]
def sample_utility(n, size_min, size_max, utility_func, utility_func_args, random_state, ub_prob=0.2, verbose=False):

  x_train, y_train, x_val, y_val = utility_func_args

  X_feature_test = []
  y_feature_test = []

  x_train = np.array(x_train)
  y_train = np.array(y_train)

  N = len(y_train)

  np.random.seed(random_state)
  
  for i in range(n):
    if verbose:
      print('{} / {}'.format(i, n))

    n_select = np.random.choice(range(size_min, size_max))

    subset_index = []

    toss = np.random.uniform()

    # With probability ub_prob, sample a class-imbalanced subset
    if toss > 1-ub_prob:
      n_per_class = int(N / 10)
      alpha = np.ones(10)
      alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
    else:
      alpha = np.random.choice(range(90, 100), size=10, replace=True)

    p = np.random.dirichlet(alpha=alpha)
    occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
    counts = np.array([np.sum(occur==i) for i in range(10)])
    
    for i in range(10):
      ind_i = np.where(np.argmax(y_train, 1)==i)[0]
      if len(ind_i) > counts[i]:
        selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
      else:
        selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
      subset_index = subset_index + list(selected_ind_i)

    subset_index = np.array(subset_index)

    y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))
    X_feature_test.append( subset_index )

  return X_feature_test, y_feature_test


def sample_utility_veryub(n, size_min, size_max, utility_func, utility_func_args, random_state, ub_prob=0.2, verbose=False):
  x_train, y_train, x_val, y_val = utility_func_args

  X_feature_test = []
  y_feature_test = []

  x_train = np.array(x_train)
  y_train = np.array(y_train)

  N = len(y_train)

  np.random.seed(random_state)
  
  for i in range(n):
    if verbose:
      print('{} / {}'.format(i, n))

    n_select = np.random.choice(range(size_min, size_max))

    if n_select > 0:
      subset_index = []

      toss = np.random.uniform()

      # With probability ub_prob, sample a class-imbalanced subset
      if toss > 1-ub_prob:
        alpha = np.random.choice(range(1, 100), size=10, replace=True)
      else:
        alpha = np.random.choice(range(90, 100), size=10, replace=True)

      p = np.random.dirichlet(alpha=alpha)
      occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
      counts = np.array([np.sum(occur==i) for i in range(10)])
      
      for i in range(10):
        ind_i = np.where(np.argmax(y_train, 1)==i)[0]
        if len(ind_i) > counts[i]:
          selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
        else:
          selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
        subset_index = subset_index + list(selected_ind_i)

      subset_index = np.array(subset_index)

      y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))
      X_feature_test.append( subset_index )
    else:
      y_feature_test.append(0.1)
      X_feature_test.append( np.array([]) )

  return X_feature_test, y_feature_test



"""### Deep Sets"""
class DeepSet(nn.Module):

    def __init__(self, in_features, set_features=128, hidden_ext=128, hidden_reg=128):
        super(DeepSet, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_ext, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(hidden_ext, hidden_ext, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(hidden_ext, set_features, bias=False)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, hidden_reg, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(hidden_reg, hidden_reg, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(hidden_reg, int(hidden_reg/2), bias=False),
            nn.ELU(inplace=True)
        )

        self.linear = nn.Linear(int(hidden_reg/2), 1)
        self.sigmoid = nn.Sigmoid()
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'


class DeepSet_cifar(nn.Module):

    def __init__(self, in_features, set_features=512):
        super(DeepSet_cifar, self).__init__()
        self.in_features = in_features
        self.out_features = set_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, set_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(set_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)
        
        
    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, "reset_parameters", None)
            if callable(reset_op):
                reset_op()
            
    def forward(self, input):
        x = input
        x = self.feature_extractor(x)
        x = x.sum(dim=1)
        x = self.regressor(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'

"""### Deepset Utility Learning Model"""

class Utility_deepset(object):

    def __init__(self, model=None):

        """
        if model is None:
          self.model = DeepSet(in_dims).cuda()
        else:
          self.model = model.cuda()
        """

        self.model = model

        self.model.linear.bias = torch.nn.Parameter(torch.tensor([-2.1972]))
        self.model.linear.bias.requires_grad = False
        #print(self.model.linear.bias)
        self.model.cuda()
        #print(self.model.linear.bias)
        
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss(reduction='sum')
        
    # train_data: x_train_few
    # train_set: (X_feature, y_feature)
    # valid_set: (X_feature_test, y_feature_test)
    def fit(self, train_data, train_set, valid_set, n_epoch, batch_size=32, lr=1e-3):

        self.optim = optim.Adam(self.model.parameters(), lr)

        #scheduler = StepLR(self.optim, step_size=10, gamma=0.1)
        scheduler = MultiStepLR(self.optim, milestones=[10,15], gamma=0.1)

        train_data = copy.deepcopy(train_data)
        N = train_data.shape[0]
        k = train_data.shape[1]

        X_feature, y_feature = train_set
        X_feature_test, y_feature_test = valid_set
        train_size = len(y_feature)

        for epoch in range(n_epoch):

          # Shuffle training utility samples
          ind = np.arange(train_size, dtype=int)
          np.random.shuffle(ind)
          X_feature = [X_feature[i] for i in ind]
          y_feature = y_feature[ind]

          train_loss = 0
          start_ind = 0

          for j in range(train_size//batch_size):
            start_ind = j*batch_size
            batch_X, batch_y = [], []
            for i in range(start_ind, min(start_ind+batch_size, train_size)):

              b = np.zeros((N, k))
              if len(X_feature[i]) > 0:
                selected_train_data = train_data[X_feature[i]]
                b[:len(X_feature[i])] = selected_train_data

              batch_X.append( b )
              batch_y.append( [y_feature[i]] )

            batch_X = np.stack(batch_X)
            batch_X, batch_y = torch.FloatTensor(batch_X).cuda(), torch.FloatTensor(batch_y).cuda()

            self.optim.zero_grad()
            y_pred = self.model(batch_X)
            loss = self.l2(y_pred, batch_y)
            loss_val = np.asscalar(loss.data.cpu().numpy())
            loss.backward()
            self.optim.step()
            train_loss += loss_val
          train_loss /= train_size
          test_loss = self.evaluate(train_data, valid_set)
          scheduler.step()
          print('Epoch %s Train Loss %s Test Loss %s' % (epoch, train_loss, test_loss))
          # print(self.model.linear.bias)
    
    def evaluate(self, train_data, valid_set):

        N, k = train_data.shape
        X_feature_test, y_feature_test = valid_set

        test_size = len(y_feature_test)
        test_loss = 0

        for i in range(test_size):

            b = np.zeros((N, k))
            if len(X_feature_test[i]) > 0:
              selected_train_data = train_data[X_feature_test[i]]
              b[:len(X_feature_test[i])] = selected_train_data

            batch_X, batch_y = torch.FloatTensor(b).cuda(), torch.FloatTensor(y_feature_test[i:i+1]).cuda()
            batch_X, batch_y = batch_X.reshape((1, N, k)), batch_y.reshape((1, 1))
            y_pred = self.model(batch_X)

            loss = self.l2(y_pred, batch_y)
            loss_val = np.asscalar(loss.data.cpu().numpy())
            test_loss += loss_val
        test_loss /= test_size
        return test_loss





def array_to_lst(X_feature):

  if type(X_feature) == list:
    return X_feature

  X_feature = list(X_feature)
  for i in range(len(X_feature)):
    X_feature[i] = X_feature[i].nonzero()[0]
  return X_feature


def findMostValuableSample_deepset_greedy(model, unlabeled_set, target_size):

  model = model.cuda()

  N, input_dim = unlabeled_set.shape
  k = target_size

  selected_subset = np.zeros(N)
  selected_rank = []
  selected_data = np.zeros((N, input_dim))

  for i in range(k):
    print(i)
    maxIndex, maxVal = -1, -1
    prevUtility = model(torch.FloatTensor(selected_data.reshape((1, N, input_dim))).cuda())
    searchRange = np.where(selected_subset == 0)[0]
    for j in searchRange:
      selected_subset[j] = 1
      selected_data[j] = unlabeled_set[j]
      utility = model(torch.FloatTensor(selected_data.reshape((1, N, input_dim))).cuda())
      selected_subset[j] = 0
      selected_data[j] = np.zeros(input_dim)
      if utility - prevUtility > maxVal:
        maxIndex = j
        maxVal = utility - prevUtility
    selected_subset[maxIndex] = 1
    selected_rank.append(maxIndex)
    selected_data[maxIndex] = unlabeled_set[maxIndex]

  return selected_subset, selected_rank, selected_data


def findMostValuableSample_deepset_stochasticgreedy(model, unlabeled_set, target_size, epsilon, seed, verbose=False, debug=False, label=None):

  model = model.cuda()

  N, input_dim = unlabeled_set.shape
  k = target_size

  selected_subset = np.zeros(N)
  selected_rank = []

  R = int((N/k)*np.log(1/epsilon))
  
  print('Sample Size R={}'.format(R))

  if debug:
    R = 10
    print('Sample Size R={}'.format(R))

  np.random.seed(seed)
  selected_data = np.zeros((N, input_dim))

  for i in range(k):
    if verbose: print(i)
    
    maxIndex, maxVal = -1, -1

    prevUtility = model(torch.FloatTensor(selected_data.reshape((1, N, input_dim))).cuda())
    searchRange = np.where(selected_subset == 0)[0]

    if debug:
      print('Step {}, prevUtility={}'.format(i, prevUtility))

    if R < len(searchRange):
      searchRange = np.random.choice(searchRange, size=R, replace=False)

    for j in searchRange:
      selected_subset[j] = 1
      selected_data[j] = unlabeled_set[j]
      utility = model(torch.FloatTensor(selected_data.reshape((1, N, input_dim))).cuda())
      selected_subset[j] = 0
      selected_data[j] = np.zeros(input_dim)
      gain = (utility - prevUtility).cpu().detach().numpy()[0][0]

      if gain > maxVal:
        maxIndex = j
        maxVal = gain

      if debug:
        print('  Gain from {} is {}, label={}'.format(j, gain, label[j]))
        print('  maxIndex={}, maxVal={}, labelMaxIndex={}'.format(maxIndex, maxVal, label[maxIndex]))
      
    selected_subset[maxIndex] = 1
    selected_rank.append(maxIndex)
    selected_data[maxIndex] = unlabeled_set[maxIndex]

  return selected_subset, selected_rank, selected_data

