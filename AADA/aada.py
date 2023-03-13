import argparse
import os
import random
import shutil
import time, pickle
import warnings
import math
import sys

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

import dataloader as dl
import pdb
from PIL import Image
import torch.utils.data as data
# from tensorboardX import SummaryWriter
import scipy.stats as stats


os.environ['CUDA_VISIBLE_DEVICES'] = '4' 

model_names = sorted(name for name in models.__dict__
  if name.islower() and not name.startswith("__")
  and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--dataset', default='', type=str, metavar='DB',
          help='dataset name')
parser.add_argument('-arch', '--arch', default='convnet', type=str, metavar='NET',
          help='network architecture (resnet101)')
parser.add_argument('--ndim', default=500, type=int, 
          help='print output embedding dimension (default: 500)')
parser.add_argument('--nclasses', default=10, type=int, 
          help='print number of output classes (default: 12)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
          help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
          help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
          help='manual epoch number (useful on restarts)')
parser.add_argument('-s2t', '--src2tgt-rate', default=1, type=float,
          help='source to target rate (default: 1)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
          metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.0002, type=float,
          metavar='LR', help='initial learning rate')
parser.add_argument('--optimizer', default='adam', type=str,
          metavar='OPT', help='stochastic optimizer')
parser.add_argument('--augmentation-list-src', default='pf,s,if,is,io', type=str,
          metavar='AUGSRC', help='augmentation list, standardize, horizontal flip, intensity flip, intensity scale, intensity offset')
parser.add_argument('--augmentation-list-tgt', default='s,if,is,io', type=str,
          metavar='AUGTGT', help='augmentation list, standardize, horizontal flip, intensity flip, intensity scale, intensity offset')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
          help='momentum')
parser.add_argument('--eps', default=1e-8, type=float, metavar='EPS',
          help='epsilon for adam optimizer')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
          metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--print-freq', '-p', default=10000, type=int,
          metavar='N', help='print frequency (default: 50)')
parser.add_argument('--prefix', default='./', type=str, metavar='SAVEPATH',
          help='path to save (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
          help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
          help='evaluate model on validation set')
parser.add_argument('-es', '--evaluate-set', type=str, default='lfw,cfp_ff,cfp_fp', 
          help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
          help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
          help='seed for initializing training. ')
parser.add_argument('-rdsrc', '--reg-disc-src', dest='reg_disc_src', default=1.0, type=float,
          help='regularizer for discriminator (source)')
parser.add_argument('-rdtgt', '--reg-disc-tgt', dest='reg_disc_tgt', default=1.0, type=float,
          help='regularizer for discriminator (source)')
parser.add_argument('-rd', '--reg_disc', dest='reg_disc', default=0.1, type=float,
          help='regularizer for discriminator (source)')
parser.add_argument('-rent', '--reg-ent', dest='reg_ent', default=0, type=float,
          help='regularizer for discriminator (source)')
parser.add_argument('-sym', '--do-sym', dest='do_sym', action='store_true',
          help='do symmetrized domain adversarial loss')

parser.add_argument('--num_Tl', default=1000, type=int,
          help='# Tl')
parser.add_argument('--seed_for_Tl', default=1, type=int,
          help='seed for selecting Tl. ')
parser.add_argument('--load_model', type=str, default='None', 
          help='load pre-trained model (for getting discriminator scores)')
parser.add_argument('--ac', type=str, default='None', 
          help='active learning method')
parser.add_argument('--run_name', type=str, default='', 
          help='folder name')
parser.add_argument('--use_weight', type=str, default='False',
          help='Use weight for loss if classes are imbalanced')
parser.add_argument('--num_rounds', default=1, type=int,
          help='# rounds for selecting samples. Each iteration select #num_Tl samples.')
parser.add_argument('--thres', default=0.8, type=float,
          help='Initial selected threshold')
parser.add_argument('--train_from_scratch', dest='train_from_scratch', action='store_true',
          help='Train from the args.load_model every round')
parser.add_argument('--sample_source', dest='sample_source', action='store_true',
          help='Subsample source s.t. |S| == |T| for each epoch to balance training on visda.')
parser.add_argument('--no_pretrain', dest='no_pretrain', action='store_true',
          help='Not using ImageNet pretraining if set to True')
parser.add_argument('--lr_disc', default=0.0002, type=float,
          help='initial learning rate for discriminator')
parser.add_argument('--finetune', type=str, default='False',
          help='Use finetuning as adapting mode. Need to set load_model as the sourceonly baseline.')
parser.add_argument('--noS', dest='noS', action='store_true',
          help='Adapt Tl<-Tu only')
parser.add_argument('--topk', default=0, type=float,
          help='choose top k percentile as the pool for selecting samples')

parser.add_argument('--plot_scatter', dest='plot_scatter', action='store_true',
          help='plot dscore')

parser.add_argument('--src_ratio', default=0, type=float)
parser.add_argument('--tgt_ratio', default=0, type=float)

best_prec1 = 0


ROOT = {'visda17':'/mnt/nfs/scratch1/jcsu/datase/visDA-2017/',
        'visda18': '/mnt/nfs/scratch1/jcsu/datase/visDA-2018/openset/',
        'visda18sbg': '/mnt/nfs/scratch1/jcsu/datase/visDA-2018/openset/',
        'visda18nbg': '/mnt/nfs/scratch1/jcsu/dataset/visDA-2018/openset/'}

class EntropyLoss(nn.Module):
  def __init__(self, reduction='elementwise_mean'):
    super(EntropyLoss, self).__init__()
    self.reduction = reduction
  def forward(self, x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    if self.reduction == 'elementwise_mean':
      b = -1.0*b.sum(1).mean()
    elif self.reduction == 'sum':
      b = -1.0*b.sum(1).sum()
    else:
      b = -1.0*b.sum(1)
    return b

def main():
  global args, best_prec1
  args = parser.parse_args()

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)
    
    warnings.warn('You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.')

  if 'mnist' in args.dataset:
    from lenet_model import Augmentation, data_transformer, load_dbs, create_model
    args.nclasses = 10
  elif 'visda' in args.dataset:
    from visda_model import create_model
    args.nclasses = 12
    args.ndim = 256
    args.arch = 'resnet18'
    args.optimizer = 'sgd'
    args.weight_decay = 0
  elif 'office' in args.dataset:
    from visda_model import create_model
    from OfficeDataset import OfficeDataset
    args.nclasses = 31
    args.ndim = 256
    args.arch = 'resnet18'
    args.optimizer = 'sgd'
    args.weight_decay = 0 
  arch = '%s_ndim%d' %(args.arch, args.ndim)
  print(args)

  ## Create Dataloader
  if args.dataset in ['visda17', 'visda18', 'visda18sbg', 'visda18nbg']:
    # visDA-17, visDA-18, visDA-18 without background class
    tf_train = transforms.Compose([
      transforms.Resize([256,256]),
      transforms.RandomCrop(224),
      transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_test = transforms.Compose([
      transforms.Resize([256,256]),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if args.dataset == 'visda17':
      root = ROOT[args.dataset]
      data_path = [os.path.join(root, 'train'), os.path.join(root, 'validation')]
      txt_path = [os.path.join(root, 'train/image_list.txt'), os.path.join(root, 'validation/image_list.txt')]
      lmdb_path = [os.path.join(root, 'train.lmdb'), os.path.join(root, 'validation.lmdb')]


  else:
    # Load MNIST and SVHN
    dataset_src, dataset_tgt = args.dataset.split('2')
    train_src, _, _ = load_dbs(root=None, dataset=dataset_src, transform=None, augmentation_list=args.augmentation_list_src)
    train_tgt, test_tgt, train_tgt_noaug = load_dbs(root=None, dataset=dataset_tgt, transform=None, augmentation_list=args.augmentation_list_tgt)  
    if args.src_ratio>0:
      selected_id = np.arange(int(len(train_src)*args.src_ratio))
      train_src = data.Subset(train_src, selected_id)
      print('after getting subset:',len(train_src))
    elif args.tgt_ratio>0:
      selected_id = np.arange(int(len(train_tgt)*args.tgt_ratio))
      train_tgt = data.Subset(train_tgt, selected_id)
      train_tgt_noaug = data.Subset(train_tgt_noaug, selected_id)
      print('after getting subset:',len(train_tgt))
  # retain Tl for k-center
  train_tgt_noaug_selected = None

  #NOTE If num_rounds == 1, it's DANN case when we only do 1 round with random selection
  start_epoch = 0 if args.num_rounds == 0 else 1
  for round_id in range(start_epoch, args.num_rounds+1):
    ## Create model
    model = create_model(arch=args.arch, ndim=args.ndim, nclasses=args.nclasses, no_pretrain=args.no_pretrain)

    ## Load model
    if args.load_model != 'None':
      if args.finetune=='True':
        print("=> using initial sourceonly model'{}'".format(args.load_model))
        for i in range(len(model)): model[i].load_state_dict(torch.load(args.load_model)['state_dict'][i])
      else:
        print("Train model from scratch")
    
    # define loss function (criterion) and optimizer
    criterion = [nn.CrossEntropyLoss(reduction='elementwise_mean').cuda(), nn.CrossEntropyLoss(reduction='elementwise_mean').cuda(), EntropyLoss(reduction='elementwise_mean').cuda()]
    if args.optimizer.lower() == 'adam':
      optimizer = [torch.optim.Adam(model[i].parameters(), args.lr, eps=args.eps, weight_decay=args.weight_decay) for i in range(len(model)-1)] + \
                  [torch.optim.Adam(model[-1].parameters(), args.lr_disc, eps=args.eps, weight_decay=args.weight_decay)]
    elif args.optimizer.lower() == 'sgd':
      optimizer = [torch.optim.SGD(model[i].parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for i in range(len(model)-1)] + \
                  [torch.optim.SGD(model[-1].parameters(), args.lr_disc, momentum=args.momentum, weight_decay=args.weight_decay)]

    # define savepath
    save_path = '%s/%s/round_%i/'%(args.prefix, args.run_name, round_id)
    print(save_path)
    if not os.path.exists(save_path):
      os.makedirs(save_path)

    # define tensorboard save path
    log_dir = save_path+'log'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # optionally resume from a checkpoint
    if args.resume:
      if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = 0
        if 'best_prec1' in checkpoint:
          best_prec1 = checkpoint['best_prec1']
        for i in range(len(model)): model[i].load_state_dict(checkpoint['state_dict'][i])
        for i in range(len(optimizer)): optimizer[i].load_state_dict(checkpoint['optimizer'][i])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
      else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if not args.sample_source:
      print('USING all source!!!')
      src_loader = torch.utils.data.DataLoader(train_src, batch_size=min(len(train_src),int(args.src2tgt_rate*args.batch_size)), shuffle=True, drop_last=True)
    tgt_loader = torch.utils.data.DataLoader(train_tgt, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_tgt, batch_size=args.batch_size, shuffle=False, drop_last=False)

    '''
    ##NOTE Training
    for epoch in range(args.start_epoch, args.epochs):
      if args.sample_source:
        selected_src_id = np.random.choice(len(train_src), len(train_tgt), replace=False)
        train_src_selected = data.Subset(train_src, selected_src_id)
        print('#(S+Tl) sampled:',len(train_src_selected))
        src_loader = torch.utils.data.DataLoader(train_src_selected, batch_size=min(len(train_src_selected),int(args.src2tgt_rate*args.batch_size)), shuffle=True, drop_last=True)
      # train for one epoch
      losses_cls, losses_ent, top1, top1_disc, losses_disc_src, losses_disc_tgt = train([src_loader, tgt_loader], model, criterion, optimizer, epoch)

      # evaluate on validation set
      filename = save_path + 'checkpoint_%04d.pth.tar' %(epoch+1)
      bestname = save_path + 'model_best.pth.tar'
      val_meter = None
      if args.evaluate and (epoch+1)%10==0:
        print('len of test_tgt',len(test_tgt))
        print('len of test_loader',len(test_loader))
        print('Val after epoch',epoch+1)
        prec1, val_meter, _, _, losses, _, _ = validate(test_loader, model, criterion, meter=val_meter)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)  
        if (epoch+1)==args.epochs:
          save_checkpoint({
            'epoch': epoch+1,
            'arch': arch,
            'best_prec1': best_prec1,
            'val_meter': val_meter,
            'state_dict': [model[i].state_dict() for i in range(len(model))],
            'optimizer' : [optimizer[i].state_dict() for i in range(len(optimizer))],
          }, is_best, filename, bestname)
      elif False:#(epoch+1)==args.epochs:
        if round_id==1:
          train_tgt_subset = None
        save_checkpoint({
          'epoch': epoch+1,
          'arch': arch,
          'state_dict': [model[i].state_dict() for i in range(len(model))],
          'optimizer' : [optimizer[i].state_dict() for i in range(len(optimizer))],
        }, False, filename)
    '''
    # import pdb; pdb.set_trace()
    # Create data loader for first round
    if round_id == 1:
      # use the source only or dann model for selecting samples in the first round
      # prev_model_path = args.load_model
      prev_model_path = '%s/%s/round_1/checkpoint_0060.pth.tar'%(args.prefix, args.run_name)

    else:
      # prev_model_path = '%s/%s/round_%i/checkpoint_%04d.pth.tar'%(args.prefix, args.run_name, round_id-1, args.epochs)
      print('FALSE!!!')

    ## Select samples
    if round_id > 0:
      if args.ac != 'None':
        ## Load model from previous round (or first round if train_from_scratch is True)
        model_prev = create_model(arch=args.arch, ndim=args.ndim, nclasses=args.nclasses)
        print("Load model '{}' from previous round for getting scores".format(prev_model_path))
        checkpoint = torch.load(prev_model_path)
        for i in range(len(model_prev)): model_prev[i].load_state_dict(checkpoint['state_dict'][i])
        
        ## Get discriminator scores (if ac is not None) from load_model

        import pdb; pdb.set_trace()
        #NOTE train_tgt_noaug is a dataset
        train_tgt_noaug = pickle.load(open('./samples/mnist_unlabel2000.data', 'rb'))
        selected_id = select_samples(model_prev, train_tgt_noaug, criterion, args.seed_for_Tl+round_id*10, args.num_Tl, args.plot_scatter, train_tgt_noaug_selected)
        pdb.set_trace()
        pickle.dump(selected_id, open('./rank/svhn2mnist_1000.rank', 'wb'))

      print('selected id:',selected_id)
      # print('selected labels:',[train_tgt_noaug[i][1] for i in selected_id])
      print('selected labels:',[train_tgt_noaug[1][i] for i in selected_id])
    
    '''
    if round_id >0 or args.num_Tl > 0:
      ## Remove Tl from train_tgt
      train_tgt_subset = data.Subset(train_tgt, selected_id)
      train_tgt = data.Subset(train_tgt, np.delete(np.array(range(len(train_tgt))),selected_id))
      # Keep Tl without augmentation for select_samples
      train_tgt_noaug = data.Subset(train_tgt_noaug, np.delete(np.array(range(len(train_tgt_noaug))),selected_id))

      ## Combine S and Tl into train_src
      if not args.sample_source and args.finetune=='False':
        ## adversarial training case, combine S and Tl if using whole S.
        ## o/w subsample S and combine in each epoch.
        if args.noS and round_id==1:
          # Discard S in the first round
          train_src = train_tgt_subset
        else:
          train_src = data.ConcatDataset([train_src, train_tgt_subset])
      elif args.finetune=='True':
        if round_id == 1:
          train_src = train_tgt_subset
        else:
          train_src = data.ConcatDataset([train_src, train_tgt_subset])

    print('#(S+Tl):',len(train_src))
    print('#(Tu):',len(train_tgt))
    '''

    ## Create dataloader
    
  # writer.export_scalars_to_json(os.path.join(args.prefix, args.run_name, 'all_scalars.json'))
  # writer.close()

def get_loader(train_tgt):
  x_unlabel = train_tgt[0]
  y_unlabel = train_tgt[1]
  tensor_x_test, tensor_y_test = torch.Tensor(x_unlabel).cuda(), torch.Tensor(y_unlabel.float()).cuda()
  test_dataset = torch.utils.data.TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, drop_last=False)

  return test_loader


def select_samples(model, train_tgt, criterion, seed_for_Tl, num_Tl, plot_scatter=False, train_tgt_noaug_selected=None):
  
  # test_loader = torch.utils.data.DataLoader(train_tgt, batch_size=128, shuffle=False, drop_last=False)
  test_loader = get_loader(train_tgt)
  import pdb; pdb.set_trace()
  
  
  get_feature = (args.ac == 'kmeans') or (args.ac == 'diversity') or (args.ac == 'kcenter')
  prec1, val_meter, all_disc_scores, all_cls_scores, _, all_features, all_output = validate(test_loader, model, criterion, \
      meter=None, get_disc_output=True, get_cls_output=True, get_feature=get_feature, get_output=False)
  
  all_cls_entropy = stats.entropy(all_cls_scores.transpose())
  pseudo_labels = np.argmax(all_cls_scores,axis=1)

  ## Plot histograms and scatter plots for discriminator scores and entropy
  if plot_scatter:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    for i in range(10):
      idx_ = np.where(pseudo_labels==i)
      plt.hist(all_disc_scores[:,1][idx_],bins=20,range=(0,1))
      plt.savefig('hist_pseudo_'+str(i)+'.png')
      plt.close()
    
    plt.hist(all_disc_scores[:,1],bins=20,range=(0,1))
    plt.savefig('dscore_hist.png')
    plt.close()
    plt.hist(all_cls_entropy,bins=162,range=(all_cls_entropy.min(),all_cls_entropy.max()))
    plt.savefig('entropy_hist.png')
    plt.close()
    plt.scatter(all_disc_scores[:,1],all_cls_entropy,s=1.0,alpha=0.5)
    plt.xlabel('discriminzator score')
    plt.ylabel('entropy')
    plt.savefig('dscore_entropy_scatter.png')
    plt.close()
    all_disc_scores[:,1] *= all_cls_entropy
    plt.hist(all_disc_scores[:,1],bins=200,range=(all_disc_scores[:,1].min(),all_disc_scores[:,1].max()))
    plt.savefig('entropy_times_dscore_hist.png')
    plt.close()

  if 'dscore_only' in args.ac:
    final_score = all_disc_scores[:,1]
  elif 'entropy_only' in args.ac:
    final_score = all_cls_entropy
  elif 'dscore_times_entropy' in args.ac:
    final_score = all_disc_scores[:,1]*all_cls_entropy
  elif 'dscore_plus_entropy' in args.ac:
    final_score = np.exp(np.log(all_disc_scores[:,1])+all_cls_entropy)
  elif 'importance_only' in args.ac:
    final_score = all_disc_scores[:,1]/all_disc_scores[:,0]
  elif 'importance_plus_entropy' in args.ac:
    final_score = all_disc_scores[:,1]/all_disc_scores[:,0]+all_cls_entropy
  elif 'importance_times_entropy' in args.ac:
    print("TRUE!!!!!!")
    final_score = all_disc_scores[:,1]/all_disc_scores[:,0]*all_cls_entropy #NOTE

  #NOTE
  if num_Tl > len(train_tgt[1]):
    num_Tl = len(train_tgt[1])

  if 'equalcls' in args.ac:
    num_Tl_cls = int(num_Tl/args.nclasses)
    np.random.seed(seed=seed_for_Tl)
    for i in range(args.nclasses):

      ## Select from top 5% instead
      thres = -np.sort(-final_score)[int(len(final_score)/20)]
      filtered_list = np.where(final_score>thres)[0]
      print('final threshold: %.2f'%thres)

      print('#scores > thres: '+str(len(filtered_list)))
      print('final threshold: %.2f'%thres)
      selected_id_cls = np.random.choice(len(filtered_list), num_Tl_cls, replace=False, \
                        p=(final_score[filtered_list])/np.sum(final_score[filtered_list]))
      if i==0:
        selected_id = filtered_list[selected_id_cls]
      else:
        selected_id = np.append(selected_id,filtered_list[selected_id_cls])

  elif args.ac == 'kmeans':
    ## K-means
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances
    kmeans = KMeans(n_clusters=num_Tl, random_state=0).fit(all_features)
    temp = pairwise_distances(all_features, kmeans.cluster_centers_)
    for i in range(num_Tl):
      center_id = np.argmin(temp[np.where(kmeans.labels_==i)[0],i])
      if i==0:
        selected_id = np.where(kmeans.labels_==i)[0][center_id]
      else:
        selected_id = np.append(selected_id, np.where(kmeans.labels_==i)[0][center_id])

  elif args.ac == 'kcenter':
    from k_center import kCenterGreedy
    if train_tgt_noaug_selected is not None:
      # first round, no Tl selected yet
      test_loader_selected = torch.utils.data.DataLoader(train_tgt_noaug_selected, batch_size=128, shuffle=False, drop_last=False)
      _, _, _, _, _, all_features_selected, _ = validate(test_loader_selected, model, criterion, \
        meter=None, get_disc_output=True, get_cls_output=True, get_feature=get_feature)
      already_selected = [i for i in range(len(train_tgt), len(train_tgt)+len(train_tgt_noaug_selected))]
      all_features = np.concatenate((all_features,all_features_selected))
      solver = kCenterGreedy(all_features, seed=seed_for_Tl)
      selected_id = solver.select_batch_(already_selected, num_Tl)
    else:
      solver = kCenterGreedy(all_features, seed=seed_for_Tl)
      selected_id = solver.select_batch_([], num_Tl)

  elif args.ac == 'diversity':
    ## In each round, for each Tu, compute the average distance (dot product of feature from Gf(x) ) to Tl, 
    ## and select images with larger distance.
    if train_tgt_noaug_selected is not None:
      # first round, no Tl selected yet
      test_loader_selected = torch.utils.data.DataLoader(train_tgt_noaug_selected, batch_size=128, shuffle=False, drop_last=False)
      _, _, _, _, _, all_features_selected, _ = validate(test_loader_selected, model, criterion, \
        meter=None, get_disc_output=True, get_cls_output=True, get_feature=get_feature)
      from sklearn.metrics.pairwise import pairwise_distances
      temp = pairwise_distances(all_features, all_features_selected)
      index = np.argsort(-np.mean(temp,axis=1))
      selected_id = index[:args.num_Tl]
    else:
      # random sampling for the first round
      np.random.seed(seed=args.seed_for_Tl)
      index = np.random.permutation(len(train_tgt))
      selected_id = index[:args.num_Tl]
    print('selected labels in this reound:',[train_tgt[selected_id[i]][1] for i in range(len(selected_id))])

  elif args.ac == 'secondbest':
    max_pred = np.max(all_cls_scores,axis=1)
    all_cls_scores[np.arange(0,all_cls_scores.shape[0]),np.argmax(all_cls_scores,axis=1)] = np.nan
    second_max_pred = np.nanmax(all_cls_scores,axis=1)
    final_score = max_pred - second_max_pred
    index = np.argsort(final_score)
    selected_id = index[:args.num_Tl]

  elif 'nothres' in args.ac:
    print('USING nothres!!!')
    index = np.argsort(-final_score)
    selected_id = index[:args.num_Tl]

  else:
    np.random.seed(seed=seed_for_Tl)
    
    if args.topk > 0:
      ## Select from top ?%
      if 'mnist' in args.dataset:
        # select 10 from top 2% of 60k = 1200
        assert(args.topk==2)
      # elif 'office' in args.dataset:
        # select 50 from top 10% of 1800 = 180, or top 5 = 90
      elif 'visda' in args.dataset:
        # select 10 from top 2% of 3600 = 72
        assert(args.topk==2)
      thres = -np.sort(-final_score)[int(len(final_score)*args.topk/100)]
      filtered_list = np.where(final_score>thres)[0]
      print('len(filtered_list):', len(filtered_list))
      print('final threshold of using top-k: %.2f'%thres)

    else:
      ## Old code for using threshold
      thres_ = args.thres
      filtered_list = np.where(final_score>thres_)[0]
      while len(filtered_list)<num_Tl:
        thres_ -= 0.1
        filtered_list = np.where(final_score>thres_)[0]
      print('#scores > thres: '+str(len(filtered_list)))
      print('final threshold: %.2f'%thres_)
      
    sigmoid_score = final_score[filtered_list]

    selected_id = np.random.choice(len(filtered_list), num_Tl, replace=False, \
                  p=(sigmoid_score)/np.sum(sigmoid_score))
    print('selected sigmoid scores:',sigmoid_score[selected_id])
    selected_id = filtered_list[selected_id]

  if 'importance' in args.ac or 'entropy' in args.ac:
    print('selected scores:',final_score[selected_id])
    print('selected pseudo labels:',pseudo_labels[selected_id])

  return selected_id

def train(data_loader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses_disc_src = AverageMeter()
  losses_disc_tgt = AverageMeter()
  losses_cls = AverageMeter()
  losses_ent = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  top1_disc = AverageMeter()

  # model, optimizer, criterion
  embed, cls, disc = model
  optim_embed, optim_cls, optim_disc = optimizer
  criterion_disc, criterion_cls, criterion_ent = criterion

  # adjust learning rate
  adjust_learning_rate(optim_embed, epoch)
  adjust_learning_rate(optim_cls, epoch)
  adjust_learning_rate(optim_disc, epoch)

  # switch to train mode
  embed.train(), cls.train(), disc.train()

  # parameters
  batch_size_src = int(args.batch_size*args.src2tgt_rate)
  batch_size_tgt = args.batch_size
  reg_disc_src = args.reg_disc_src
  reg_disc_tgt = args.reg_disc_tgt
  reg_disc = args.reg_disc
  reg_ent = args.reg_ent
  do_sym = args.do_sym

  n_batch = min(len(data_loader[0]), len(data_loader[1]))
  nclasses = args.nclasses

  # generate target vector
  end = time.time()
  nBatch = min(len(data_loader[0]), len(data_loader[1]))

  for (i, (input_src, target_src)), (_, (input_tgt, _)) in zip(enumerate(data_loader[0]), enumerate(data_loader[1])):
    domain_src = torch.zeros(input_src.shape[0]).long().fill_(0).cuda()
    domain_tgt = torch.zeros(input_tgt.shape[0]).long().fill_(1).cuda()

    # measure data loading time
    data_time.update(time.time() - end)
    target_src = target_src.cuda()

    # update discriminator
    optim_disc.zero_grad()

    with torch.no_grad():
      output_src = embed(input_src)
    output_disc_src = disc(output_src)
    loss_disc_src = criterion_disc(output_disc_src, domain_src.fill_(0))
    losses_disc_src.update(loss_disc_src.item(), batch_size_src)
    loss_disc_src *= reg_disc_src
    loss_disc_src.backward()
    with torch.no_grad():
      output_tgt = embed(input_tgt)
    output_disc_tgt = disc(output_tgt)
    loss_disc_tgt = criterion_disc(output_disc_tgt, domain_tgt.fill_(1))
    losses_disc_tgt.update(loss_disc_tgt.item(), batch_size_tgt)
    loss_disc_tgt *= reg_disc_tgt
    loss_disc_tgt.backward()

    prec_src = accuracy(output_disc_src, domain_src.fill_(0))
    top1_disc.update(prec_src[0][0], batch_size_src)
    prec_tgt = accuracy(output_disc_tgt, domain_tgt.fill_(1))
    top1_disc.update(prec_tgt[0][0], batch_size_tgt)
    # if top1_disc.avg.cpu().numpy()<80.0:
    optim_disc.step()

    # update classifier and embedding networks
    optim_cls.zero_grad()
    optim_embed.zero_grad()

    # source (classification only)
    output_src = embed(input_src)
    output_cls_src = cls(output_src)
    loss_cls = criterion_cls(output_cls_src, target_src)
    losses_cls.update(loss_cls.item(), batch_size_src)
    prec1, prec5 = accuracy(output_cls_src[:,:nclasses], target_src, topk=(1, 5))
    top1.update(prec1[0], batch_size_src)
    top5.update(prec5[0], batch_size_src)
    if do_sym:
      output_disc_src = disc(output_src)
      loss_disc_src = criterion_disc(output_disc_src, domain_src.fill_(1))
      loss_cls += reg_disc * loss_disc_src
    loss_cls.backward()

    # target (discriminator, entropy)
    output_tgt = embed(input_tgt)
    output_disc_tgt = disc(output_tgt)
    if reg_disc > 0:
      # otherwise just finetuning, not training this
      loss_disc_tgt = criterion_disc(output_disc_tgt, domain_tgt.fill_(0))

      if reg_ent > 0:
        output_cls_tgt = cls(output_tgt)
        loss_ent = criterion_ent(output_cls_tgt)
        losses_ent.update(loss_ent.item(), batch_size_tgt)
        if reg_disc > 0:
          loss = reg_disc * loss_disc_tgt + (reg_ent*reg_disc) * loss_ent
        else:
          loss = reg_disc * loss_disc_tgt + reg_ent * loss_ent
      else:
        loss = reg_disc * loss_disc_tgt
      loss.backward()

    optim_cls.step()
    optim_embed.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if (i % args.print_freq == 0 and i > 0) or (i+1 == nBatch):
      print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'CLS {loss_cls.val:.3f} ({loss_cls.avg:.3f})\t'
          'ENT {loss_ent.val:.3f} ({loss_ent.avg:.3f})\t'
          'D(S) {loss_src.val:.3f} ({loss_src.avg:.3f})\t'
          'D(T) {loss_tgt.val:.3f} ({loss_tgt.avg:.3f})\t'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
          'Prec_Disc@1 {top1_disc.val:.3f} ({top1_disc.avg:.3f})'.format(
           epoch+1, i+1, n_batch, batch_time=batch_time,
           data_time=data_time, loss_cls=losses_cls, loss_ent=losses_ent, 
           loss_src=losses_disc_src, loss_tgt=losses_disc_tgt, top1=top1, top1_disc=top1_disc))
  return losses_cls.avg, losses_ent.avg, top1.avg, top1_disc.avg, losses_disc_src.avg, losses_disc_tgt.avg


def validate(val_loader, model, criterion, meter=None, get_disc_output=False, \
             get_cls_output=False, get_feature=False, get_output=False):
  if meter is None:
    meter = HistoryMeter()
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  embed, cls, disc = model
  embed.eval(), cls.eval()

  cor_per_cls, tot_per_cls = {}, {}
  nclasses = args.nclasses
  for i in range(nclasses): cor_per_cls[i] = 0
  for i in range(nclasses): tot_per_cls[i] = 0

  all_disc_scores = None
  all_cls_scores = None
  all_features = None
  all_output = None
  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      target = target.cuda().long()

      # compute output
      if get_feature:
        feat_output = embed(input.cuda())
        if all_features is None:
          all_features = feat_output.data.cpu().numpy()
        else:
          all_features = np.concatenate((all_features,feat_output))

      if get_disc_output:
        disc_output = disc(embed(input.cuda()))
        disc_scores = nn.functional.softmax(disc_output, dim=1).cpu().numpy()
        if all_disc_scores is None:
          all_disc_scores = disc_scores
        else:
          all_disc_scores = np.concatenate((all_disc_scores,disc_scores))
        # source=0, target=1

      output = cls(embed(input.cuda()))
      if get_output:
        if all_output is None:
          all_output = output.data.cpu().numpy()
        else:
          all_output = np.concatenate((all_output,output))
      if get_cls_output:
        cls_scores = nn.functional.softmax(output, dim=1).cpu().numpy()
        if all_cls_scores is None:
          all_cls_scores = cls_scores
        else:
          all_cls_scores = np.concatenate((all_cls_scores,cls_scores))
      _, pred = output.topk(5, 1, True, True)
      for j in range(len(target)):
        tot_per_cls[int(target[j])] += 1
        if int(target[j]) == int(pred[j][0]):
          cor_per_cls[int(target[j])] += 1
      loss = criterion[0](output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output[:,:nclasses], target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1[0], input.size(0))
      top5.update(prec5[0], input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0 and i > 0:
        print('Test: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
             i+1, len(val_loader), batch_time=batch_time, loss=losses,
             top1=top1, top5=top5))

    acc = []
    for i in sorted(cor_per_cls.keys()):
      if tot_per_cls[i] > 0:
        acc += [cor_per_cls[i]/tot_per_cls[i]]
      else:
        acc += [0.0]
    acc_str = ''
    for i in range(len(acc)):
      acc_str += '%.2f ' %(100.0*acc[i])
    acc_str += '| %.2f' %(100.0*np.mean(acc))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print(' * Per class accuracy: %s' %acc_str)
    meter.update(top1.avg, acc)

  return top1.avg, meter, all_disc_scores, all_cls_scores, losses.avg, all_features, all_output


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', bestname=None):
  torch.save(state, filename)
  print('save file: ' + filename)
  if is_best:
    shutil.copyfile(filename, bestname)


class HistoryMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.acc = []
    self.acc_per_cls = []

  def update(self, acc, acc_per_cls):
    self.acc.append(acc)
    self.acc_per_cls.append(acc_per_cls)
    

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  if 'mnist' in args.dataset:
    if epoch >= 40:
      lr = 0.00005
    elif epoch >= 20:
      lr = 0.0001
    else:
      lr = args.lr
  elif 'visda' in args.dataset:
    if epoch >= 50:
      lr = args.lr/100.0
    elif epoch >= 30:
      lr = args.lr/10.0
    else:
      lr = args.lr
  elif 'office' in args.dataset:
    if epoch >= 50:
      lr = args.lr/100.0
    elif epoch >= 30:
      lr = args.lr/10.0
    else:
      lr = args.lr

  print('learning rate = %g' %lr)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
  main()
