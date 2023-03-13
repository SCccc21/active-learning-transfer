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
import copy, os
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.optimize as op 
import seaborn as sns
import pickle

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

from distil.utils.data_handler import DataHandler_Points
from distil.active_learning_strategies import glister, FASS, BADGE


from deepsets import *
import time


def getSelectedAcc(rank, target_size, x_train, y_train, x_val, y_val, utilityFunc, interval=100):
  if utilityFunc is None:
    return None
  ret = np.zeros(int(target_size/interval))
  for i in range(1, int(target_size/interval)+1):
    ret[i-1] = utilityFunc(x_train[rank[:interval*(i)]], y_train[rank[:interval*(i)]].float(), x_val, y_val.float())
  return ret


def getSelectedAcc_ft(adda_net_file, rank, target_size, x_train, y_train, x_val, y_val, interval=100):
  ret = np.zeros(int(target_size/interval))
  for i in range(1, int(target_size/interval)+1):
    ret[i-1] = data_to_acc_finetune(adda_net_file, x_train[rank[:interval*(i)]], y_train[rank[:interval*(i)]], x_val, y_val)
  return ret




# attack_type = sys.argv[1]
trans_type = 'visda_new'

coll_dir = './rank-collections/'
os.makedirs(coll_dir, exist_ok=True)
rank_coll_dir = coll_dir + trans_type + '.rankcoll'
acc_coll_dir = coll_dir + trans_type + '.acccoll'
# import pdb; pdb.set_trace()

# Choose GPU ID
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 


if trans_type == 'visda_new':
  from utils_visda import *
  n_data = 10000
  target_size_tot = 5000
  
  n_data_deepset = 1000
  n_block = int(n_data/n_data_deepset) # 10
  target_size = int(target_size_tot/n_block) # single block
  n_epoch = 15
  n_set = 128
  n_hext = 128
  n_hreg = 128
  LR = 1e-4
  n_cls = 12
  interval = 500

  sample_dir = './samples/visda/'
  x_train_few, y_train_few =  pickle.load(open(sample_dir + '/visda_src_train500.data', 'rb'))
  x_val, y_val = pickle.load(open(sample_dir + '/visda_src_test500.data', 'rb'))
  

  model_dir = './models/visda'
  visda_dir = '/home/chensi/AFN/vanilla/Visda2017/HAFN/code/snapshot_ds'
  deepset_dir = model_dir + '/deepset_128_128_128_10_47.state_dict'
  deepset_optimal_dir = model_dir + '/visda_optimal_128_128_128_3_23.state_dict'
  adda_net_file = [visda_dir+'/VisDA_HAFN_resnet101_netG_1.1_10.pth', visda_dir+'/VisDA_HAFN_resnet101_netF_1.1_10.pth']

  
  #lc_rank, sv_rank, loo_rank, tmc_rank, gshap_rank, inf_rank, tracin_rank, knn_rank = pickle.load(
  #  open(rank_coll_dir+'Backdoor_CIFAR_N1000.rankcoll', 'rb'))

  
  func_data_to_acc = torch_visda_resnet_data_to_acc_multiple

  # load feature extractor

  netG = ResBase101().cuda()
  netF = ResClassifier(class_num=12, extract=True).cuda()
  netG.load_state_dict(torch.load(adda_net_file[0]))
  netG.eval()
  netF.load_state_dict(torch.load(adda_net_file[1]))
  netF.eval()

  

  x_train_few_resNetFeature = featureExtract(x_train_few, netG, netF)
  # val feature will be used when perform AL selection
  x_val_few_resNetFeature = featureExtract(x_val, netG, netF)


  # unlabel feature
  tgt_x_unlabel_few, tgt_y_unlabel_few = pickle.load(open(sample_dir+'/visda_unlabel10000.data', 'rb'))
  tgt_x_test, tgt_y_test = pickle.load(open(sample_dir + '/visda_tgt_test2000.data', 'rb'))
  tgt_x_unlabel_resNetFeature = featureExtract(tgt_x_unlabel_few, netG, netF)
  tgt_x_unlabel_cnnFeature = tgt_x_unlabel_resNetFeature




# Load DeepSet
ds = DeepSet(x_train_few_resNetFeature.shape[1], set_features=n_set, hidden_ext=n_hext, hidden_reg=n_hreg)
deepset_model = Utility_deepset(model=ds)
deepset_model.model.load_state_dict(torch.load(deepset_dir))
deepset_model.model.eval()

ds_optimal = DeepSet(tgt_x_unlabel_cnnFeature.shape[1], set_features=n_set, hidden_ext=n_hext, hidden_reg=n_hreg)
deepset_model_optimal = Utility_deepset(model=ds_optimal)
deepset_model_optimal.model.load_state_dict(torch.load(deepset_optimal_dir))
deepset_model_optimal.model.eval()


eps = 1e-3

deepset_rank_coll = np.zeros((6, target_size_tot))
random_rank_coll = np.zeros((6, target_size_tot))
optimal_rank_coll = np.zeros((6, target_size_tot))
fass_rank_coll = np.zeros((6, target_size_tot))
badge_rank_coll = np.zeros((6, target_size_tot))
glister_rank_coll = np.zeros((6, target_size_tot))

acc_coll = {}
ftacc_coll = {}

acc_coll['deepset'] = []
acc_coll['random'] = []
acc_coll['optimal'] = []
acc_coll['fass'] = []
acc_coll['badge'] = []
acc_coll['glister'] = []

ftacc_coll['deepset'] = []
ftacc_coll['random'] = []
ftacc_coll['optimal'] = []
ftacc_coll['fass'] = []
ftacc_coll['badge'] = []
ftacc_coll['glister'] = []



# import pdb; pdb.set_trace()
net_al = torch_encoder_logistic_data_to_net(x_train_few_resNetFeature, y_train_few.float(), x_val_few_resNetFeature, y_val.float())
for select_seed in range(6):

  deepset_rank_matrix = np.zeros((n_block, target_size))
  optimal_rank_matrix = np.zeros((n_block, target_size))
  fass_rank_matrix = np.zeros((n_block, target_size))
  badge_rank_matrix = np.zeros((n_block, target_size))
  glister_rank_matrix = np.zeros((n_block, target_size))

  random_perm = np.random.permutation(range(n_data))

  
  for i in range(n_block):
    print("Block:", i)

    random_ind = random_perm[n_data_deepset*(i):n_data_deepset*(i+1)]
    start_time = time.time()
    _, deepset_rank_small, _ = findMostValuableSample_deepset_stochasticgreedy(deepset_model.model, tgt_x_unlabel_resNetFeature[random_ind], target_size, epsilon=eps, seed=select_seed)
    print(time.time()-start_time)
    _, optimal_rank_small, _ = findMostValuableSample_deepset_stochasticgreedy(deepset_model_optimal.model, tgt_x_unlabel_cnnFeature[random_ind], target_size, epsilon=eps, seed=select_seed)
    
    
    deepset_rank_matrix[i] = random_ind[deepset_rank_small]
    optimal_rank_matrix[i] = random_ind[optimal_rank_small]




  print('FASS select...')
  strategy_args = {'batch_size':target_size_tot, 'submod':'facility_location', 'selection_type':'PerClass'}
  strategy = FASS(x_train_few_resNetFeature, y_train_few, tgt_x_unlabel_resNetFeature, net_al.cuda(), DataHandler_Points, n_cls, strategy_args)
  fass_rank = strategy.select(target_size_tot)

  print('BADGE select...')
  strategy_args = {'batch_size' : target_size_tot}
  strategy = BADGE(x_train_few_resNetFeature, y_train_few, tgt_x_unlabel_resNetFeature, net_al.cuda(), DataHandler_Points, n_cls, strategy_args)
  badge_rank = strategy.select(target_size_tot)

  # print('GLISTER select...')
  # strategy_args = {'batch_size' : target_size_tot, 'lr':float(0.001)}
  # strategy = glister.GLISTER(x_train_few_resNetFeature, y_train_few, tgt_x_unlabel_resNetFeature, net_al.cuda(), DataHandler_Points, n_cls, strategy_args, 
  #                           valid=True, X_val=x_val_few_resNetFeature, Y_val=y_val)
  # glister_rank = strategy.select(target_size_tot)



  # get the whole rank
  deepset_rank = ((deepset_rank_matrix.T).reshape(-1)).astype(int)
  optimal_rank = ((optimal_rank_matrix.T).reshape(-1)).astype(int)

  deepset_rank_coll[select_seed] = deepset_rank
  optimal_rank_coll[select_seed] = optimal_rank
  fass_rank_coll[select_seed] = fass_rank
  badge_rank_coll[select_seed] = badge_rank
  # glister_rank_coll[select_seed] = glister_rank

  random_rank = np.random.permutation(range(target_size_tot))
  random_rank_coll[select_seed] = random_rank

  

  # import pdb; pdb.set_trace()
  # get acc
  deepset_acc = getSelectedAcc(deepset_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, func_data_to_acc, interval=interval)
  optimal_acc = getSelectedAcc(optimal_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, func_data_to_acc, interval=interval)
  fass_acc = getSelectedAcc(fass_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, func_data_to_acc, interval=interval)
  badge_acc = getSelectedAcc(badge_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, func_data_to_acc, interval=interval)
  # glister_acc = getSelectedAcc(glister_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, func_data_to_acc, interval=interval)
  random_acc = getSelectedAcc(random_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, func_data_to_acc, interval=interval)


  acc_coll['deepset'].append(deepset_acc)
  acc_coll['optimal'].append(optimal_acc)
  acc_coll['fass'].append(fass_acc)
  acc_coll['badge'].append(badge_acc)
  # acc_coll['glister'].append(glister_acc)
  acc_coll['random'].append(random_acc)



  # FINETUNE
  deepset_acc = getSelectedAcc_ft(adda_net_file, deepset_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, interval=interval)
  optimal_acc = getSelectedAcc_ft(adda_net_file, optimal_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, interval=interval)
  fass_acc = getSelectedAcc_ft(adda_net_file, fass_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, interval=interval)
  badge_acc = getSelectedAcc_ft(adda_net_file, badge_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, interval=interval)
  # glister_acc = getSelectedAcc_ft(adda_net_file, glister_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, interval=interval)
  random_acc = getSelectedAcc_ft(adda_net_file, random_rank, target_size_tot, tgt_x_unlabel_few, tgt_y_unlabel_few, tgt_x_test, tgt_y_test, interval=interval)

  ftacc_coll['deepset'].append(deepset_acc)
  ftacc_coll['optimal'].append(optimal_acc)
  ftacc_coll['fass'].append(fass_acc)
  ftacc_coll['badge'].append(badge_acc)
  # ftacc_coll['glister'].append(glister_acc)
  ftacc_coll['random'].append(random_acc)



  pickle.dump([deepset_rank_coll, random_rank_coll, optimal_rank_coll, fass_rank_coll, badge_rank_coll, glister_rank_coll], open(rank_coll_dir, 'wb'))
  print("Rank of seed {} saved!".format(select_seed))
  pickle.dump([acc_coll, ftacc_coll], open(acc_coll_dir, 'wb'))
  print("Finetune Acc of seed {} saved!".format(select_seed))
