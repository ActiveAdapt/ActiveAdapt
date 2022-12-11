# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='indoor', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=2, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="/data/ztc/adaptation/Experiment/model",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_false')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


def MIR(training_t_set_name,loop_count, old_data_name,retrive_num,virtual_finetune_model,load_name,method,picked_class_list):
  setup_seed(3)
  args = parse_args()

  #print('Called with args:')
  #print(args)
  args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
  #print('Using config:')
  #pprint.pprint(cfg)
  

  ########################### get virtual model
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = True
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.dataset+"_2007_"+old_data_name+str(loop_count-1))
  train_size = len(roidb)  # add flipped         image_index*2
  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0)
  num_images = len(imdb.image_index)
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if True:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if True:
    cfg.CUDA = True
  
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()


  fasterRCNN.create_architecture()
  print("load checkpoint %s" % (load_name))
  checkpoint=torch.load(load_name)
  fasterRCNN.load_state_dict({k:v for k,v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})
  if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
 
  if True:
    fasterRCNN.cuda()

  fasterRCNN.train()
  virtual_finetune_model.train()
  loss_temp = 0
  v_loss_temp = 0
  '''
  data_iter = iter(dataloader)
  memory_list=[]
  class_i_dict={k:[] for k in xrange(1, imdb.num_classes)}
  for i in range(num_images):
    data = next(data_iter)
    with torch.no_grad():
      gt_boxes.resize_(data[2].size()).copy_(data[2])
    class_lab=round(gt_boxes[0][0,-1].detach().cpu().item())
    class_i_dict[class_lab].append(i)
  for k in xrange(1, imdb.num_classes):
    memory_list.extend(random.sample(class_i_dict[k],round(50./5)))

  del data_iter
  '''
  class_i_dict={k:[] for k in xrange(1, imdb.num_classes)}

  data_iter = iter(dataloader)  
  loss_distance_list=[]
  image_name_list=[]
  image_class_list=[]   
  for i in range(num_images):
    data = next(data_iter)
    if True:#i in memory_list:#
      with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
      im_name=data[4][0].split('/')[-1][:-4]
      #print(im_name)
      class_lab=round(gt_boxes[0][0,-1].detach().cpu().item())
      if len(class_i_dict[class_lab])>=70:#50 for 5 class
        continue
      else:
        class_i_dict[class_lab].append(im_name)
      
        fasterRCNN.zero_grad()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        virtual_finetune_model.zero_grad()
        v_rois, v_cls_prob, v_bbox_pred, \
        v_rpn_loss_cls, v_rpn_loss_box, \
        v_RCNN_loss_cls, v_RCNN_loss_bbox, \
        v_rois_label = virtual_finetune_model(im_data, im_info, gt_boxes, num_boxes)

        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        loss_temp += loss.item()
        v_loss = v_rpn_loss_cls.mean() + v_rpn_loss_box.mean() + v_RCNN_loss_cls.mean() + v_RCNN_loss_bbox.mean()
        v_loss_temp += v_loss.item()

        loss_distance_list.append(v_loss_temp-loss_temp)
        image_name_list.append(im_name)#imdb.image_path_at(i).split('/')[-1][:-4]) 
        image_class_list.append(round(gt_boxes[0][0,-1].detach().cpu().item()))
  
        loss_temp = 0
        v_loss_temp = 0
  
  final_retrive_names=[]
  for j in xrange(1, imdb.num_classes):
    pick_num=round((retrive_num/7)-picked_class_list.count(j))#divided by 5 for five class
    #print(pick_num)
    inds = torch.nonzero(torch.Tensor(image_class_list)==j).view(-1)
    #print(inds,len(inds))
    class_loss_distance_list=[loss_distance_list[p] for p in inds]
    sorted_loss=sorted(range(len(class_loss_distance_list)), key=lambda k:class_loss_distance_list[k], reverse=True)
    #print(sorted_loss,len(sorted_loss))
    class_image_name_list=[image_name_list[p] for p in inds]
    retrive_names = [class_image_name_list[i] for i in sorted_loss][0:pick_num]
    #print(retrive_names,len(retrive_names))
    final_retrive_names.extend(retrive_names)
  print(len(final_retrive_names))
  '''
  pick_num=retrive_num-len(picked_class_list)
  sorted_loss=sorted(range(len(loss_distance_list)), key=lambda k:loss_distance_list[k], reverse=True)
  final_retrive_names = [image_name_list[i] for i in sorted_loss][0:pick_num]#random.sample(image_name_list,retrive_num)#
  '''
  final_training_t_set_name="final_training_t_CL_"+method
  final_training_t_set=open("/root/code/faster-rcnn.pytorch/data/"+args.dataset+"/VOC2007/ImageSets/Main/"+final_training_t_set_name+str(loop_count)+".txt","w")
  with open("/root/code/faster-rcnn.pytorch/data/"+args.dataset+"/VOC2007/ImageSets/Main/"+training_t_set_name+str(loop_count)+".txt","r") as f:
    for line in f.readlines():
      final_training_t_set.write(line)      
  for line in final_retrive_names:
    final_training_t_set.write(line+"\n")
  final_training_t_set.close()
       
  return final_training_t_set_name
