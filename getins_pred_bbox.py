import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from collections import Counter

from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import save_net, load_net, vis_detections

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def getins_pred_bbox(fasterRCNN,im_data,im_info,gt_boxes,num_boxes,cfg,args,incom_imdb,data,i,empty_array,all_boxes):
    judge=True
    score_f = 0.
    thresh = 0.05

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if args.class_agnostic:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas = box_deltas.view(1, -1, 4)
          else:
              box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              box_deltas = box_deltas.view(1, -1, 4 * len(incom_imdb.classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    
    pred_boxes /= data[1][0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    vis=True
    if vis:
        im = cv2.imread(incom_imdb.image_path_at(i))
        #print(incom_imdb.image_path_at(i))
        im2show = np.copy(im)
    for j in xrange(1, incom_imdb.num_classes):
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          if args.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
          else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
            
          all_boxes[j][i] = cls_dets.cpu().numpy()
        else:
          all_boxes[j][i] = empty_array
    
    show_boxes = {j:[] for j in xrange(1, incom_imdb.num_classes)}
    max_score=0
    ins_pred_bbox=None
    for j in xrange(1, incom_imdb.num_classes):
        #print('class j',j)
        if vis:
            im2show = vis_detections(im2show, incom_imdb.classes[j], all_boxes[j][i], 0.5)
            vis_score=[all_boxes[j][i][d,-1] for d in range(np.minimum(10, all_boxes[j][i].shape[0])) if all_boxes[j][i][d,-1]>0.5]
            show_boxes[j]=[all_boxes[j][i][d,:4] for d in range(np.minimum(10, all_boxes[j][i].shape[0])) if all_boxes[j][i][d,-1]>0.5]
            #print(vis_score)
            #print(show_boxes)
            if len(vis_score)!=0:
                temp_max_score=max(vis_score)
                if temp_max_score>max_score:
                    max_score=temp_max_score
                    ins_pred_bbox=show_boxes[j][vis_score.index(temp_max_score)].astype(int)
                #print(max_score)
                #print(ins_pred_bbox)
    #exit()         
    return max_score,ins_pred_bbox
