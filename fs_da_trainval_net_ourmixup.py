# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

#from roi_data_layer.roidb import combined_roidb
#from roi_data_layer.roibatchLoader import roibatchLoader

from roi_da_data_layer.roidb import combined_roidb
from roi_da_data_layer.roibatchLoader import roibatchLoader


from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.fs_da_faster_rcnn.vgg16_ourmixup import vgg16
from model.fs_da_faster_rcnn.resnet_ourmixup import resnet

from evaluate_model import evaluate_model

#from model.da_faster_rcnn.vgg16 import vgg16
#from model.da_faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='cityscape', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="/data/ztc/adaptation/Experiment/da_model",
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
                      action='store_true')
  parser.add_argument('--s_bs', dest='s_batch_size',
                      help='source_batch_size',
                      default=3, type=int)
  parser.add_argument('--t_bs', dest='t_batch_size',
                      help='target_batch_size',
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
                      default=0.002, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=6, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--lamda', dest='lamda',
                      help='DA loss param',
                      default=0.6, type=float)


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
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
  
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      print('loading our dataset...........')
      args.imdb_name = "voc_2007_train"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "cityscape":
      print('loading our dataset...........')
      args.s_imdb_name = "cityscape_2007_train_s"
      args.t_imdb_name = "cityscape_2007_train_t_fs"
      args.s_imdbtest_name="cityscape_2007_test_s"
      args.t_imdbtest_name="cityscape_2007_test_t"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "lemur":
      print('loading our dataset...........')
      args.s_imdb_name = "lemur_2007_train_s"
      args.t_imdb_name = "lemur_2007_train_t_fs"
      args.s_imdbtest_name="lemur_2007_test_s"
      args.t_imdbtest_name="lemur_2007_test_t"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "indoor":
      print('loading indoor dataset.........')
      args.s_imdb_name = "indoor_2007_train_mixup_s"
      args.t_imdb_name = "indoor_2007_train_mixup_t_10fs"
      args.s_imdbtest_name = "indoor_2007_test_mixup_s"
      args.t_imdbtest_name = "indoor_2007_test_mixup_t"
      #args.all_imdb_name="indoor_2007_train_all_fs"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "cleanlemur":
      print('loading our dataset...........')
      args.s_imdb_name = "cleanlemur_2007_train_s"
      args.t_imdb_name = "cleanlemur_2007_toy_t_10fs"#"cleanlemur_2007_train_t_fs_exper1"#fs,5fs,10fs,fs_exper1
      args.s_imdbtest_name="cleanlemur_2007_test_s"
      args.t_imdbtest_name="cleanlemur_2007_test_t"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  setup_seed(3)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.s_imdb_name)
  s_train_size = len(s_roidb)  # add flipped         image_index*2

  t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)
  t_train_size = len(t_roidb)  # add flipped         image_index*2

  print('source {:d} target {:d} roidb entries'.format(len(s_roidb),len(t_roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  s_sampler_batch = sampler(s_train_size, args.s_batch_size)
  t_sampler_batch=sampler(t_train_size,args.t_batch_size)

  s_dataset = roibatchLoader(s_roidb, s_ratio_list, s_ratio_index, args.s_batch_size, \
                           s_imdb.num_classes, training=True)

  s_dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=args.s_batch_size,
                            sampler=s_sampler_batch, num_workers=args.num_workers)


  t_dataset=roibatchLoader(t_roidb, t_ratio_list, t_ratio_index, args.t_batch_size, \
                           t_imdb.num_classes, training=True)

  t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=args.t_batch_size,
                                           sampler=t_sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  need_backprop = torch.FloatTensor(1)

  tgt_im_data = torch.FloatTensor(1)
  tgt_im_info = torch.FloatTensor(1)
  tgt_num_boxes = torch.LongTensor(1)
  tgt_gt_boxes = torch.FloatTensor(1)
  tgt_need_backprop = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda:
      im_data = im_data.cuda()
      im_info = im_info.cuda()
      num_boxes = num_boxes.cuda()
      gt_boxes = gt_boxes.cuda()
      need_backprop = need_backprop.cuda()

      tgt_im_data = tgt_im_data.cuda()
      tgt_im_info = tgt_im_info.cuda()
      tgt_num_boxes = tgt_num_boxes.cuda()
      tgt_gt_boxes = tgt_gt_boxes.cuda()
      tgt_need_backprop = tgt_need_backprop.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  need_backprop = Variable(need_backprop)

  tgt_im_data = Variable(tgt_im_data)
  tgt_im_info = Variable(tgt_im_info)
  tgt_num_boxes = Variable(tgt_num_boxes)
  tgt_gt_boxes = Variable(tgt_gt_boxes)
  tgt_need_backprop = Variable(tgt_need_backprop)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(s_imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(s_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(s_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  #finetune_epoch=10
  #load_name = "/data/ztc/adaptation/Experiment/da_model/vgg16/cleanlemur/train_cleanlemur_ours_epoch"+str(finetune_epoch)+"_lambda0.5_lr0.001.pth"
  #print("load checkpoint %s" % (load_name))
  #checkpoint=torch.load(load_name)
  #fasterRCNN.load_state_dict({k:v for k,v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})
  #if 'pooling_mode' in checkpoint.keys():
  #    cfg.POOLING_MODE = checkpoint['pooling_mode']

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

  if args.optimizer == "adam":
    #lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    print("Using cuda for faster rcnn network")
    fasterRCNN.cuda()

  iters_per_epoch = int(s_train_size/args.s_batch_size)#int(s_train_size / args.s_batch_size)
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.max_epochs)
  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    new_epoch_flag=0
    fasterRCNN.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(s_dataloader)
    tgt_data_iter=iter(t_dataloader)
    tgt_iter = [tgt_i+(round(random.randint(0,iters_per_epoch-t_train_size)/t_train_size)*t_train_size) for tgt_i in range(t_train_size)]#random.randint(0,iters_per_epoch)
    print(tgt_iter)
    for step in range(iters_per_epoch):
      #print(step)
      new_epoch_flag = new_epoch_flag + 1
      data = next(data_iter)
      try:
        tgt_data=next(tgt_data_iter)
      except:
        del tgt_data_iter
        tgt_data_iter = iter(t_dataloader)
        tgt_data = next(tgt_data_iter)
      
      with torch.no_grad():
        im_data.resize_(data[0].size()).copy_(data[0])
        im_info.resize_(data[1].size()).copy_(data[1])
        gt_boxes.resize_(data[2].size()).copy_(data[2])
        num_boxes.resize_(data[3].size()).copy_(data[3])
        need_backprop.resize_(data[4].size()).copy_(data[4])
      
        tgt_im_data.resize_(tgt_data[0].size()).copy_(tgt_data[0])
        tgt_im_info.resize_(tgt_data[1].size()).copy_(tgt_data[1])
        tgt_gt_boxes.resize_(tgt_data[2].size()).copy_(tgt_data[2])
        tgt_num_boxes.resize_(tgt_data[3].size()).copy_(tgt_data[3])
        tgt_need_backprop.resize_(tgt_data[4].size()).copy_(tgt_data[4])
      

      """   faster-rcnn loss + DA loss for source and   DA loss for target    """
      fasterRCNN.zero_grad()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label,\
      mix_rois, mix_cls_prob, mix_bbox_pred, \
      mix_rpn_loss_cls, mix_rpn_loss_box, \
      mix_RCNN_loss_cls, mix_RCNN_loss_bbox, \
      mix_rois_label,\
      tgt_rois, tgt_cls_prob, tgt_bbox_pred, \
      tgt_rpn_loss_cls, tgt_rpn_loss_box, \
      tgt_RCNN_loss_cls, tgt_RCNN_loss_bbox, \
      tgt_rois_label,\
      DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_img_loss_cls,tgt_DA_ins_loss_cls=\
      fasterRCNN(im_data, im_info, gt_boxes, num_boxes,need_backprop,
                     tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop,iters_per_epoch,new_epoch_flag)

      #print(rpn_loss_cls.mean())
      #print(step, tgt_iter)
      if False:#(step in tgt_iter):
        tgt_loss= tgt_rpn_loss_cls.mean() + tgt_rpn_loss_box.mean()+ tgt_RCNN_loss_cls.mean() + tgt_RCNN_loss_bbox.mean()
        print("train tgt")
      else:
        tgt_loss=torch.tensor(0).float().cuda()
      
      loss = tgt_loss+3*(rpn_loss_cls.mean() + rpn_loss_box.mean() + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean())+3*(mix_rpn_loss_cls.mean() + mix_rpn_loss_box.mean() + mix_RCNN_loss_cls.mean() + mix_RCNN_loss_bbox.mean())#+args.lamda*(DA_img_loss_cls.mean()+tgt_DA_img_loss_cls.mean()+DA_ins_loss_cls.mean()+tgt_DA_ins_loss_cls.mean())#+DA_cst_loss.mean()+tgt_DA_cst_loss.mean())
      loss_temp += loss.item()
      

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
          
          tgt_loss_rpn_cls = tgt_rpn_loss_cls.mean().item()
          tgt_loss_rpn_box = tgt_rpn_loss_box.mean().item()
          tgt_loss_rcnn_cls = tgt_RCNN_loss_cls.mean().item()
          tgt_loss_rcnn_box = tgt_RCNN_loss_bbox.mean().item()
          tgt_fg_cnt = torch.sum(tgt_rois_label.data.ne(0))
          tgt_bg_cnt = tgt_rois_label.data.numel() - tgt_fg_cnt
          

        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
         
          tgt_loss_rpn_cls = tgt_rpn_loss_cls.item()
          tgt_loss_rpn_box = tgt_rpn_loss_box.item()
          tgt_loss_rcnn_cls = tgt_RCNN_loss_cls.item()
          tgt_loss_rcnn_box = tgt_RCNN_loss_bbox.item()

          #loss_DA_img_cls=args.lamda*(DA_img_loss_cls.item()+tgt_DA_img_loss_cls.item())/2
          #loss_DA_ins_cls = args.lamda * (DA_ins_loss_cls.item() + tgt_DA_ins_loss_cls.item()) / 2
          #loss_DA_cst = args.lamda * (DA_cst_loss.item() + tgt_DA_cst_loss.item()) / 2
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

          tgt_fg_cnt = torch.sum(tgt_rois_label.data.ne(0))
          tgt_bg_cnt = tgt_rois_label.data.numel() - tgt_fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\ttgt_fg/tgt_bg=(%d/%d), time cost: %f" % (tgt_fg_cnt, tgt_bg_cnt, end-start))

        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f,\n\t\t\ttgt_rpn_cls: %.4f, tgt_rpn_box: %.4f, tgt_rcnn_cls: %.4f, tgt_rcnn_box %.4f,\n\t\t\timg_loss %.4f,ins_loss %.4f,,cst_loss %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,tgt_loss_rpn_cls, tgt_loss_rpn_box, tgt_loss_rcnn_cls, tgt_loss_rcnn_box,0,0,0))#,loss_DA_img_cls,loss_DA_ins_cls,0))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            'tgt_loss_rpn_cls': tgt_loss_rpn_cls,
            'tgt_loss_rpn_box': tgt_loss_rpn_box,
            'tgt_loss_rcnn_cls': tgt_loss_rcnn_cls,
            'tgt_loss_rcnn_box': tgt_loss_rcnn_box,


          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()
    
    #scheduler.step()
    #if epoch==args.max_epochs:
    save_name = os.path.join(output_dir, 'indoor_ourmixup_10fs_epoch{}.pth'.format(epoch))
    #save_name = os.path.join(output_dir, 'train_cleanlemur_source+ourmixup_lambda{}_lr{}_beta0.2_all_1_offline_epoch{}.pth'.format(args.lamda, args.lr,epoch))#train_cleanlemur_ours+ourmixup_lambda{}_lr{}_beta0.2_epoch{}args.lamda, args.lr,
    save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
    print('save model: {}'.format(save_name))
    #evaluate_model(fasterRCNN, -1)
  if args.use_tfboard:
    logger.close()
