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

def judgeimage(fasterRCNN,im_data,im_info,gt_boxes,num_boxes,cfg,args,incom_imdb,data,i,empty_array,all_boxes):
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
    mean_score=[]
    class_pred = []
    show_boxes = {j:[] for j in xrange(1, incom_imdb.num_classes)}
    for j in xrange(1, incom_imdb.num_classes):
        #print('class j',j)
        if vis:
            im2show = vis_detections(im2show, incom_imdb.classes[j], all_boxes[j][i], 0.5)
            vis_score=[all_boxes[j][i][d,-1] for d in range(np.minimum(10, all_boxes[j][i].shape[0])) if all_boxes[j][i][d,-1]>0.5]
            real_score=[all_boxes[j][i][d,-1] for d in range(np.minimum(10, all_boxes[j][i].shape[0])) if all_boxes[j][i][d,-1]>0.]
            show_boxes[j]=[all_boxes[j][i][d,:4] for d in range(np.minimum(10, all_boxes[j][i].shape[0])) if all_boxes[j][i][d,-1]>0.5]
            mean_score.extend(real_score)
            class_pred.extend([j for d in vis_score])
            #print("class",incom_imdb.classes[j],"score:", vis_score,"real score",real_score)
    #print("mean score:", np.mean(mean_score))
    if len(mean_score)!=0:
        score_f=np.mean(mean_score)

    if num_boxes[0] == len(class_pred):      
        if dict(Counter(gt_boxes[0,:num_boxes[0],-1].tolist()))==dict(Counter(class_pred)):
            match_box={box_k:[] for box_k in xrange(1, incom_imdb.num_classes)}
            for bo in range(num_boxes[0]):
                box=gt_boxes[0,bo,:4]
                gt_xmin=box[0]/im_info[0][-1]
                gt_ymin=box[1]/im_info[0][-1]
                gt_xmax=box[2]/im_info[0][-1]
                gt_ymax=box[3]/im_info[0][-1]
                #print("gt:",gt_xmin,gt_ymin,gt_xmax,gt_ymax)
                gt_area=(gt_ymax-gt_ymin)*(gt_xmax-gt_xmin)
                gt_class=gt_boxes[0,bo,-1].type(torch.int)
                find_flag=False
                area_overlap=0
                for pred_bo in range(len(show_boxes[gt_class.cpu().item()])):
                    if pred_bo in match_box[gt_class.cpu().item()]:
                        continue
                    else:
                        p_box=show_boxes[gt_class.cpu().item()][pred_bo]
                        p_xmin=p_box[0]
                        p_ymin=p_box[1]
                        p_xmax=p_box[2]
                        p_ymax=p_box[3]
                        #print("pred:",p_xmin,p_ymin,p_xmax,p_ymax)
                        p_area=(p_ymax-p_ymin)*(p_xmax-p_xmin)
                        w_overlap=min(gt_xmax,p_xmax)-max(gt_xmin,p_xmin)
                        h_overlap=min(gt_ymax,p_ymax)-max(gt_ymin,p_ymin)
                        if (w_overlap>=0) and (h_overlap>=0):
                            area_overlap=w_overlap*h_overlap
                            #print(area_overlap/p_area,area_overlap/gt_area)
                            if area_overlap/p_area>=0.35:
                                if ((gt_ymax-gt_ymin)>=(2*(gt_xmax-gt_xmin))) or ((gt_xmax-gt_xmin)>=(2*(gt_ymax-gt_ymin))):
                                    if area_overlap/gt_area>=0.25:
                                        find_flag=True
                                        match_box[gt_class.cpu().item()].append(pred_bo)
                                else:
                                    if area_overlap/gt_area>=0.35:
                                        find_flag=True
                                        match_box[gt_class.cpu().item()].append(pred_bo)
                if find_flag==False:
                    judge=False
                    #print("predict wrong!!! one GT didn't find corresponding prediction")
                    break
        else:
            judge=False
            #print("predict wrong!!! class is not correct")
    else:
        judge=False
        #print("predict wrong!!! class number is not correct")
    #print("Finish judgement")
    '''
    if vis:
        #cv2.imwrite('result.png', im2show)
        #pdb.set_trace()
        cv2.imshow('test', im2show)
        #print("show im")
        cv2.waitKey(0)
    '''
    return judge,score_f
