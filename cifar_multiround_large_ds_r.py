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

from utility_functions import *
from pubfig83_utility import *
from deepset import *


def getSelectedAcc(rank, target_size, x_train, y_train, x_val, y_val, utilityFunc, interval=50):
  if utilityFunc is None:
    return None
  ret = np.zeros(int(target_size/interval))
  for i in range(1, int(target_size/interval)+1):
    ret[i-1] = utilityFunc(x_train[rank[::-1][interval*(i):]], y_train[rank[::-1][interval*(i):]], x_val, y_val)
  return ret



attack_type = sys.argv[1]

rank_coll_dir = 'rank-collections/'



if attack_type == 'Large_Backdoor_CIFAR':
  n_data = 10000
  data_seed = 100
  target_size = int(n_data)
  
  n_data_deepset = 1000
  n_epoch = 15
  n_set = 128
  n_hext = 128
  n_hreg = 128
  LR = 1e-4

  x_train_few, y_train_few, x_val, y_val = pickle.load( open('low-quality-data/backdoor_cifar_10000_2000.data', 'rb') )

  x_poi, y_poi = pickle.load( open('low-quality-data/backdoor_cifar_val.data', 'rb') )

  n_poison = 2000

  deepset_dir = 'saved-deepset/Backdoor_CIFAR_N{}_DataSeed100_Nepoch{}_Nset{}_Next{}_Nreg{}_LR{}.state_dict'.format(
    n_data_deepset, n_epoch, n_set, n_hext, n_hreg, LR)


  #lc_rank, sv_rank, loo_rank, tmc_rank, gshap_rank, inf_rank, tracin_rank, knn_rank = pickle.load(
  #  open(rank_coll_dir+'Backdoor_CIFAR_N1000.rankcoll', 'rb'))

  acc_coll_dir = rank_coll_dir+'Large_Backdoor_CIFAR_N10000_DS.acccoll'

  func_data_to_acc = torch_cifar_smallCNN_data_to_acc_multiple
  func_data_to_atkacc = torch_cifar_smallCNN_data_to_acc_multiple


elif attack_type == 'Large_Noisy_CIFAR':
  n_data = 20000
  data_seed = 100
  target_size = int(n_data)

  # DeepSet Parameters
  n_data_deepset = 1000
  n_epoch = 3
  n_set = 128
  n_hext = 128
  n_hreg = 128
  LR = 1e-4

  n_poison = 5000

  x_train_few, y_train_few, x_val, y_val = pickle.load(open('low-quality-data/noisy_cifar_20000_5000.data', 'rb'))
  x_poi, y_poi = None, None

  deepset_dir = 'saved-deepset/Noisy_CIFAR_N{}_DataSeed300_Nepoch{}_Nset{}_Next{}_Nreg{}_LR{}.state_dict'.format(
    n_data_deepset, n_epoch, n_set, n_hext, n_hreg, LR)

  #lc_rank, sv_rank, loo_rank, tmc_rank, gshap_rank, inf_rank, tracin_rank, knn_rank = pickle.load(
  #  open(rank_coll_dir+'Backdoor_CIFAR_N1000.rankcoll', 'rb'))

  acc_coll_dir = rank_coll_dir+'Large_Noisy_CIFAR_N20000_DS.acccoll'

  func_data_to_acc = cifar_bigcnn_data_to_acc_multiple
  func_data_to_atkacc = None



print(y_train_few[:10])

# Load Feature Extractor
opt = tf.train.AdamOptimizer()
extractor = CifarExtractor.build(numChannels=3, imgRows=32, imgCols=32, numClasses=10)
extractor.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
load_status = extractor.load_weights('cifar_featureExtractor.ckpt')
lastLayerOp = K.function([extractor.layers[0].input], [extractor.layers[-5].output])

x_train_few_cnnFeature = extractFeatures(lastLayerOp, x_train_few)


# Load DeepSet
ds = DeepSet(x_train_few_cnnFeature.shape[1], 10, set_features=n_set, hidden_ext=n_hext, hidden_reg=n_hreg)
deepset_model = Utility_deepset(in_dims=x_train_few_cnnFeature.shape[1], in_classes=10, lr=LR, set_feature=n_set, model=ds)
deepset_model.model.load_state_dict(torch.load(deepset_dir))
deepset_model.model.eval()


eps = 1e-3

if len(y_train_few.shape) < 2 or y_train_few.shape[1]!=10:
  y_train_few_hot = np_utils.to_categorical(y_train_few)
else:
  y_train_few_hot = y_train_few


deepset_rank_coll = np.zeros((10, n_data))
random_rank_coll = np.zeros((10, n_data))
acc_coll = {}
atkacc_coll = {}

acc_coll['deepset'] = []
acc_coll['lc'] = []
acc_coll['sv'] = []
acc_coll['loo'] = []
acc_coll['tmc'] = []
acc_coll['gshap'] = []
acc_coll['inf'] = []
acc_coll['tracin'] = []
acc_coll['knn'] = []
acc_coll['random'] = []

atkacc_coll['deepset'] = []
atkacc_coll['lc'] = []
atkacc_coll['sv'] = []
atkacc_coll['loo'] = []
atkacc_coll['tmc'] = []
atkacc_coll['gshap'] = []
atkacc_coll['inf'] = []
atkacc_coll['tracin'] = []
atkacc_coll['knn'] = []
atkacc_coll['random'] = []


for select_seed in range(10):

  n_block = int(n_data/n_data_deepset)

  deepset_rank_matrix = np.zeros((n_block, n_data_deepset))
  random_perm = np.random.permutation(range(n_data))

  for i in range(n_block):
    
    random_ind = random_perm[n_data_deepset*(i):n_data_deepset*(i+1)] #NOTE shuffle the unlabel data

    _, deepset_rank_small, _ , _ = findMostValuableSample_deepset_stochasticgreedy(deepset_model.model, 
                                                                                   x_train_few_cnnFeature[random_ind], 
                                                                                   y_train_few_hot[random_ind], 
                                                                                   n_data_deepset, epsilon=eps, seed=select_seed)

    deepset_rank_matrix[i] = random_ind[deepset_rank_small]

  deepset_rank = ((deepset_rank_matrix.T).reshape(-1)).astype(int)
  deepset_rank_coll[select_seed] = deepset_rank

  random_rank = np.random.permutation(range(n_data))
  random_rank_coll[select_seed] = random_rank

  half_size = int(target_size / 2)
  interval = int(half_size / 10)

  
  deepset_acc = getSelectedAcc(deepset_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  """
  lc_acc = getSelectedAcc(lc_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  sv_acc = getSelectedAcc(sv_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  loo_acc = getSelectedAcc(loo_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  tmc_acc = getSelectedAcc(tmc_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  gshap_acc = getSelectedAcc(gshap_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  inf_acc = getSelectedAcc(inf_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  tracin_acc = getSelectedAcc(tracin_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  knn_acc = getSelectedAcc(knn_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)
  """
  random_acc = getSelectedAcc(random_rank, half_size, x_train_few, y_train_few, x_val, y_val, func_data_to_acc, interval)

  acc_coll['deepset'].append(deepset_acc)
  """
  acc_coll['lc'].append(lc_acc)
  acc_coll['sv'].append(sv_acc)
  acc_coll['loo'].append(loo_acc)
  acc_coll['tmc'].append(tmc_acc)
  acc_coll['gshap'].append(gshap_acc)
  acc_coll['inf'].append(inf_acc)
  acc_coll['tracin'].append(tracin_acc)
  acc_coll['knn'].append(knn_acc)
  """
  acc_coll['random'].append(random_acc)

  deepset_acc = getSelectedAcc(deepset_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  """
  lc_acc = getSelectedAcc(lc_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  sv_acc = getSelectedAcc(sv_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  loo_acc = getSelectedAcc(loo_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  tmc_acc = getSelectedAcc(tmc_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  gshap_acc = getSelectedAcc(gshap_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  inf_acc = getSelectedAcc(inf_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  tracin_acc = getSelectedAcc(tracin_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  knn_acc = getSelectedAcc(knn_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)
  """
  random_acc = getSelectedAcc(random_rank, half_size, x_train_few, y_train_few, x_poi, y_poi, func_data_to_atkacc, interval)

  atkacc_coll['deepset'].append(deepset_acc)
  """
  atkacc_coll['lc'].append(lc_acc)
  atkacc_coll['sv'].append(sv_acc)
  atkacc_coll['loo'].append(loo_acc)
  atkacc_coll['tmc'].append(tmc_acc)
  atkacc_coll['gshap'].append(gshap_acc)
  atkacc_coll['inf'].append(inf_acc)
  atkacc_coll['tracin'].append(tracin_acc)
  atkacc_coll['knn'].append(knn_acc)
  """
  atkacc_coll['random'].append(random_acc)

  pickle.dump([deepset_rank_coll, random_rank_coll, acc_coll, atkacc_coll], open(acc_coll_dir, 'wb') )

  print('save!!!')
