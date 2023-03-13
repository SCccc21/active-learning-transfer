import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import itertools
from scipy.special import comb
from cvxopt import matrix, solvers, spdiag
import copy
import pickle
import sys
import time
import os

# scikit-learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Keras / Tensorflow Libraries
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Dense, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
import tensorflow.keras.utils as np_utils
import tensorflow



def load_bigcnn_model():
    num_classes = 10
    weight_decay = 1e-3
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', name='dense'))

    opt = tf.train.AdadeltaOptimizer(learning_rate=1)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def cifar_bigcnn_data_to_acc(x_train, y_train, x_val, y_val, verbose=0):

    if len(y_train.shape) < 2 or y_train.shape[1]!=10:
      y_train = np_utils.to_categorical(y_train)

    if len(y_val.shape) < 2 or y_val.shape[1]!=10:
      y_val = np_utils.to_categorical(y_val)

    if len(np.sum(y_train, axis=0).nonzero()[0]) == 1:
        return 0.1

    model = load_bigcnn_model()
    model.fit(x_train, y_train, epochs=20, verbose=verbose, batch_size=32, validation_data=(x_val, y_val))
    acc = model.evaluate(x_val, y_val, verbose=0)[1]

    return acc

def cifar_bigcnn_data_to_acc_multiple(x_train, y_train, x_val, y_val, repeat=3, verbose=0):
    acc_lst = []
    for _ in range(repeat):
        acc = cifar_bigcnn_data_to_acc(x_train, y_train, x_val, y_val, verbose)
        acc_lst.append(acc)
    return np.mean(acc_lst)




def torch_cifar_smallCNN_checkpoison(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = SmallCNN_CIFAR().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  softmax = nn.Softmax()

  for epoch in range(10):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

  for images, labels in test_loader:
      images = Variable(images)
      outputs = net(images)
  
  return softmax(outputs.data).cpu().numpy()[0][6]


def torch_cifar_smallCNN_checkpoison_multiple(x_train, y_train, x_test, y_test, verbose=0, repeat=5):
  ret = []
  for _ in range(repeat):
    ret.append(torch_cifar_smallCNN_checkpoison(x_train, y_train, x_test, y_test))
  return np.mean(ret)




def torch_covid_smallCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)
  #y_train = y_train.reshape(y_train.size, 1)

  # criterion = torch.nn.CrossEntropyLoss()
  criterion = nn.BCEWithLogitsLoss()

  net = SmallCNN_DOGCAT().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  highest_acc = 0

  for epoch in range(25):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels.unsqueeze(1))
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          # _, predicted = torch.max(outputs.data, 1)
          predicted = torch.round( torch.sigmoid(outputs.data) )
          total += labels.size(0)
          correct += (predicted == labels.unsqueeze(1)).sum()
      accuracy = correct/total
      highest_acc = max(accuracy.item(), highest_acc)
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), 100*accuracy))
  return highest_acc


def torch_covid_smallCNN_data_to_acc_multiple(x_train, y_train, x_val, y_val, repeat=3, verbose=0):
  acc_lst = []
  for _ in range(repeat):
    acc = torch_covid_smallCNN_data_to_acc(x_train, y_train, x_val, y_val, verbose)
    acc_lst.append(acc)
  return np.mean(acc_lst)



def svm_reweighted_accuracy(A, y, predictions):

    # Count the amount of correct prediction for each race
    correct_count = np.zeros(max(A)+1)
    A_count = np.unique(A, return_counts=True)[1]

    for i in range(len(A)):
      if y[i]==predictions[i]:
        correct_count[A[i]] += 1
    return 1/max(A) * np.sum(np.divide(correct_count, A_count)) 

def compas_svm_data_to_weightedacc(x_train, y_train, x_test, y_test, a_test, verbose=0):

  if len(np.unique(y_train)) == 1:
    return 0.5

  model = SVC(C=0.1, kernel='rbf')
  model.fit(x_train, y_train)
  predictions = model.predict(x_test)
  acc = svm_reweighted_accuracy(a_test, y_test, predictions)
  return acc



def spam_lgreg_to_acc(x_train, y_train, x_val, y_val):
  if np.sum(y_train) == 0 or np.sum(y_train) == y_train.shape[0]:
    return 0.5
  model = LogisticRegression(max_iter=1000)
  model.fit(x_train, y_train)
  predicted_labels = model.predict(x_val)
  acc = accuracy_score(y_val, predicted_labels)
  return acc



def reweighted_accuracy(A, y, predictions):
    count0 = count1= 0
    nA_one = np.count_nonzero(A)
    nA_zero = len(A)-nA_one
    for i in range(len(A)):
        if y[i]==predictions[i] and A[i]==0:
            count0 += 1
        if A[i]==predictions[i] and A[i]==1:
            count1 +=1 
            
    return 0.5*((1/nA_zero)*count0 + (1/nA_one)*count1)

def adult_logistic_data_to_weightedacc(x_train, y_train, x_test, y_test, a_test, verbose=0):

  if len(np.unique(y_train)) == 1:
    return 0.5

  model = LogisticRegression(max_iter=1000)
  model.fit(x_train, y_train)
  predictions = model.predict(x_test)
  acc = reweighted_accuracy(a_test, y_test, predictions)
  return acc


# utility_func_args = [x_train, y_train, x_val, y_val]
def sample_utility(n, size_min, size_max, utility_func, utility_func_args, random_state, ub_prob=0.2, verbose=False):

  X_feature_test = []
  y_feature_test = []

  if len(utility_func_args) == 4:
    x_train, y_train, x_val, y_val = utility_func_args
  elif len(utility_func_args) == 5:
    x_train, y_train, x_val, y_val, a_val = utility_func_args

  x_train, y_train = np.array(x_train), np.array(y_train)

  N = len(y_train)

  np.random.seed(random_state)
  
  for i in range(n):
    if verbose: print('Sample {} / {}'.format(i, n))

    n_select = np.random.choice(range(size_min, size_max))

    subset_index = []

    toss = np.random.uniform()

    # With probability ub_prob, sample a class-imbalanced subset
    if toss > 1-ub_prob:
      n_per_class = int(N / 10)
      alpha = np.ones(10)
      alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
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

    else:
      subset_index = np.random.choice(range(N), n_select, replace=False)

    subset_index = np.array(subset_index)

    if len(utility_func_args) == 4:
      y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val))
    elif len(utility_func_args) == 5:
      y_feature_test.append(utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val, a_val))

    X_feature_test.append(subset_index)

  return X_feature_test, y_feature_test




class SmallCNN_CIFAR(nn.Module):
    def __init__(self):
        super(SmallCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, last=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        if last:
          return output, x
        else:
          return output

    def getFeature(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x

    def get_embedding_dim(self):
        return 84



def torch_cifar_smallCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = SmallCNN_CIFAR().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(10):
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


def torch_cifar_smallCNN_data_to_acc_multiple(x_train, y_train, x_val, y_val, repeat=3, verbose=0):
  acc_lst = []
  for _ in range(repeat):
    acc = torch_cifar_smallCNN_data_to_acc(x_train, y_train, x_val, y_val, verbose)
    acc_lst.append(acc)
  return np.mean(acc_lst)



class KerasLeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses, activation="relu", weightsPath=None):

        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)
        if K.image_data_format() == "channels_first": inputShape = (numChannels, imgRows, imgCols)

        if numChannels > 1:
          inputShape = (numChannels, imgRows, imgCols)

        model.add(Conv2D(20, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(40, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model



def mnist_cnn_data_to_acc(x_train, y_train, x_val, y_val, verbose=0):

  if len(y_train.shape)<2:
    y_train = np_utils.to_categorical(y_train, 10)

  if len(y_val.shape)<2:
    y_val = np_utils.to_categorical(y_val, 10)

  if len(np.sum(y_train, axis=0).nonzero()[0]) == 1:
    return 0.1

  model = KerasLeNet.build(numChannels=1, imgRows=28, imgCols=28, numClasses=10)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=5, verbose=verbose, batch_size=128)
  acc = model.evaluate(x_val, y_val)[1]

  return acc

def mnist_cnn_data_to_acc_multiple(x_train, y_train, x_val, y_val, repeat=3, verbose=0):
  acc_lst = []
  for _ in range(repeat):
    acc = mnist_cnn_data_to_acc(x_train, y_train, x_val, y_val, verbose)
    acc_lst.append(acc)
  return np.mean(acc_lst)




class SmallCNN_DOGCAT(nn.Module):
    def __init__(self):
        super(SmallCNN_DOGCAT, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x, last=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        if last:
          return output, x
        else:
          return output

    def getFeature(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        return x

    def get_embedding_dim(self):
        return 84


def torch_dogcat_smallCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)
  #y_train = y_train.reshape(y_train.size, 1)

  # criterion = torch.nn.CrossEntropyLoss()
  criterion = nn.BCEWithLogitsLoss()

  net = SmallCNN_DOGCAT().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(25):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels.unsqueeze(1))
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          # _, predicted = torch.max(outputs.data, 1)
          predicted = torch.round( torch.sigmoid(outputs.data) )
          total += labels.size(0)
          correct += (predicted == labels.unsqueeze(1)).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), 100*accuracy))
  return accuracy.item()


def torch_dogcat_smallCNN_data_to_acc_multiple(x_train, y_train, x_val, y_val, repeat=3, verbose=0):
  acc_lst = []
  for _ in range(repeat):
    acc = torch_dogcat_smallCNN_data_to_acc(x_train, y_train, x_val, y_val, verbose)
    acc_lst.append(acc)
  return np.mean(acc_lst)

