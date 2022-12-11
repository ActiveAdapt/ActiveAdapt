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
from time import sleep, time
import copy

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


def MIR(training_t_set_name,loop_count, old_data_name,retrive_num,virtual_finetune_model,vgg_model,method,picked_class_list):
  setup_seed(3)
  t1=time()
  model=copy.deepcopy(vgg_model)
  dataset_name="cleanlemur"#"indoor"#

  ########################### get virtual model
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = True
  imdb, roidb, ratio_list, ratio_index = combined_roidb(dataset_name+"_2007_"+old_data_name+str(loop_count-1))
  train_size = len(roidb)  # add flipped         image_index*2
  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                           imdb.num_classes, training=True)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=0)
  num_images = len(imdb.image_index)
  t2=time()
  print("!!!!!!Time to load data in MIR:"+str(t2-t1))
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

  cfg.CUDA = True
  
  model.cuda().train()
  virtual_finetune_model.train()
  loss_temp = 0
  v_loss_temp = 0
  '''
  data_iter = iter(dataloader)
  
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
  memory_list=[]
  class_i_dict={k:[] for k in xrange(1, imdb.num_classes)}

  data_iter = iter(dataloader)  
  loss_distance_list=[]
  image_name_list=[]
  image_class_list=[]   
  for i in range(num_images):
    data = next(data_iter)
    
    with torch.no_grad():
      im_data.resize_(data[0].size()).copy_(data[0])
      im_info.resize_(data[1].size()).copy_(data[1])
      gt_boxes.resize_(data[2].size()).copy_(data[2])
      num_boxes.resize_(data[3].size()).copy_(data[3])
    im_name=data[4][0].split('/')[-1][:-4]
    #print(im_name)
    class_lab=round(gt_boxes[0][0,-1].detach().cpu().item())
    if len(class_i_dict[class_lab])>=50:
      continue
    else:
      class_i_dict[class_lab].append(im_name)

      model.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = model(im_data, im_info, gt_boxes, num_boxes)

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
      image_name_list.append(im_name)#(imdb.image_path_at(i).split('/')[-1][:-4]) 
      image_class_list.append(round(gt_boxes[0][0,-1].detach().cpu().item()))
  
      loss_temp = 0
      v_loss_temp = 0
  t3=time()
  print("Time to calculate loss in MIR:"+str(t3-t2))
  final_retrive_names=[]
  for j in xrange(1, imdb.num_classes):
    pick_num=round((retrive_num/5)-picked_class_list.count(j))#indoor is 7#lemur is 5
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

  final_training_t_set_name="final_training_t_CL_"+method
  final_training_t_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+final_training_t_set_name+str(loop_count)+".txt","w")
  with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+training_t_set_name+str(loop_count)+".txt","r") as f:
    for line in f.readlines():
      final_training_t_set.write(line)      
  for line in final_retrive_names:
    final_training_t_set.write(line+"\n")
  final_training_t_set.close()
  t4=time()
  print("Time to select in MIR:"+str(t4-t3))
       
  return final_training_t_set_name
