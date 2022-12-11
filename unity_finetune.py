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
import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient




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


def finetune(vgg_model,cfg,method,training_t_set_name,loop_count,copy_model,batch_size=1,max_epoch=2,minibatch_flag=False):
  setup_seed(3)
  disp_interval=20
  net="vgg16"
  dataset_name="cleanlemur"#"indoor"#
  
  cfg.TRAIN.USE_FLIPPED = False
  cfg.USE_GPU_NMS = True
  if copy_model==True:
    model=copy.deepcopy(vgg_model)
  else:
    model=vgg_model

  if minibatch_flag==True:
    minibatch_time=2
    minibatch_num=30#40#indoor is 40#65#before lemur is 30
    dataloader_list=[]
    for m_t in range(minibatch_time):
      tmp_minibatch_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/tmp_minibatch_"+method+str(m_t)+".txt","w")
      tmp_minibatch_list=[]
      with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+training_t_set_name+str(loop_count)+".txt","r") as f:
        for line in f.readlines():
          line=line.strip('\n')
          tmp_minibatch_list.append(line) 
      picked_minibatch_list=random.sample(tmp_minibatch_list,minibatch_num)    
      for line in picked_minibatch_list:
        tmp_minibatch_set.write(line+"\n")
      tmp_minibatch_set.close()
      imdb, roidb, ratio_list, ratio_index = combined_roidb(dataset_name+"_2007_tmp_minibatch_"+method+str(m_t))
      train_size = len(roidb)  # add flipped         image_index*2
      sampler_batch = sampler(train_size, batch_size)

      tmp_dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                           imdb.num_classes, training=True)

      tmp_dataloader = torch.utils.data.DataLoader(tmp_dataset, batch_size=batch_size,
                            sampler=sampler_batch, num_workers=0)
      dataloader_list.append(tmp_dataloader)

  else:
    imdb, roidb, ratio_list, ratio_index = combined_roidb(dataset_name+"_2007_"+training_t_set_name+str(loop_count))
    train_size = len(roidb)  # add flipped         image_index*2
    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    sampler_batch = sampler(train_size, batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size, \
                           imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler_batch, num_workers=0)

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
 
  lr = 0.001
  lr_decay_gamma=0.1
  lr_decay_step=5

  params = []
  for key, value in dict(model.named_parameters()).items():
    #print(key)
    if 'base' not in key:
      if value.requires_grad:
        if 'bias' in key:
          params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  model.cuda()

  iters_per_epoch = int(train_size / batch_size)

  for epoch in range(1, max_epoch + 1):
    # setting to train mode

    model.train()
    loss_temp = 0
    start = time.time()

    if epoch % (lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, lr_decay_gamma)
        lr *= lr_decay_gamma
    if minibatch_flag==True:
      data_iter = iter(dataloader_list[0])
      iters_per_epoch=iters_per_epoch*minibatch_time
    else:
      data_iter = iter(dataloader)
    batch_count=0
    for step in range(iters_per_epoch):
      try:
        data = next(data_iter)
      except:
        del data_iter
        batch_count=batch_count+1
        data_iter = iter(dataloader_list[batch_count])
        data = next(data_iter)
      #data = next(data_iter)

      with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
      
      model.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = model(im_data, im_info, gt_boxes, num_boxes)

      loss = batch_size*(rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean())
      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      if net == "vgg16":
          clip_gradient(model, 10.)
      optimizer.step()

      if step % disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (disp_interval + 1)

        loss_rpn_cls = rpn_loss_cls.item()
        loss_rpn_box = rpn_loss_box.item()
        loss_rcnn_cls = RCNN_loss_cls.item()
        loss_rcnn_box = RCNN_loss_bbox.item()
        fg_cnt = torch.sum(rois_label.data.ne(0))
        bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (0, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

        loss_temp = 0
        start = time.time()
  return model
