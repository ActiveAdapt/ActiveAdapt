import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.rpn.generate_anchors import generate_anchors
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.ourmixup_proposal_target_layer_cascade import _ourmixup_ProposalTargetLayer
from sklearn.preprocessing import OneHotEncoder

from model.fs_da_faster_rcnn.DA import _ImageDA
from model.fs_da_faster_rcnn.DA import _InstanceDA
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.ourmixup_RCNN_proposal_target = _ourmixup_ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.split_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

        self.RCNN_imageDA = _ImageDA(self.dout_base_model*2)
        self.RCNN_instanceDA = _InstanceDA(self.n_classes)
        self.consistency_loss = torch.nn.MSELoss(size_average=False)
    
    def ourmixup_fullback(self, batch_size, im_data, im_info, gt_boxes, num_boxes, tgt_im_data,tgt_gt_boxes,tgt_num_boxes):
        mix_lambda=max(0,min(1,np.random.beta(1.5,1.5)))#random.uniform(0.7,1)#
        #mix_label_lambda = max(0,min(1,np.random.beta(0.2,0.2)))
        mix_height = im_data.size(2)
        mix_width = im_data.size(3)
        mix_img = np.zeros((batch_size, im_data.size(1), mix_height, mix_width), dtype=np.float32)
        mix_im_info = im_info.detach().clone()
        mix_gt_boxes = gt_boxes.detach().clone()
        mix_num_boxes = num_boxes.detach().clone()
        #print(mix_gt_boxes.size())#(3,50,5)add one hot space here
        mix_gt_boxes_onehot_append = torch.zeros([gt_boxes.size(0), gt_boxes.size(1),self.n_classes]).cuda()
        mix_gt_boxes = torch.cat((mix_gt_boxes,mix_gt_boxes_onehot_append),-1)
        #print(mix_gt_boxes.size())#(3,50,11)
        
        for mix_idx in range(batch_size):
            
            mix_im_info[mix_idx][2] = 1.
            
            onehot_im = self.onehot(mix_gt_boxes[mix_idx,:,4].long().view(-1,1),self.n_classes)
            mix_gt_boxes[mix_idx,:,5:] = onehot_im
            onehot_tgt = self.onehot(tgt_gt_boxes[0, :,4].long().view(-1,1),self.n_classes)
            #print(onehot_im.size(),onehot_tgt.size())#(50,6)
            mix_onehot = onehot_im.detach().clone()
            mix_img[mix_idx,:,:im_data.size(2),:im_data.size(3)] = im_data[mix_idx].detach().clone().cpu().numpy()*mix_lambda
            tgt_patch_box_list = []
            if tgt_num_boxes[0]>num_boxes[mix_idx]: # past random tgt patch to im box one by one             
                for patch in range(num_boxes[mix_idx]):
                   tgt_patch_idx = random.randint(0,tgt_num_boxes[0]-1)
                   tgt_patch_box = tgt_gt_boxes[0,tgt_patch_idx,:]
                   tgt_patch = tgt_im_data[0,:,tgt_patch_box[1].type(torch.int):tgt_patch_box[3].type(torch.int),tgt_patch_box[0].type(torch.int):tgt_patch_box[2].type(torch.int)].unsqueeze(0)
                   
                   resized_tgt_patch = F.interpolate(tgt_patch, size=((gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int)),(gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int)))).squeeze(0)
                   #print(im_data[mix_idx].size(),mix_gt_boxes[mix_idx,num_boxes[mix_idx]-1,:],patch)
                   #print(gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int),gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int))
                   #print(resized_tgt_patch.size())
                   #print(gt_boxes[mix_idx,patch,1].type(torch.int)-gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int)-gt_boxes[mix_idx,patch,2].type(torch.int))
                   #print(mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)].shape,(resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)).shape)
                   mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)] += resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)
                   tgt_patch_box_list.append(tgt_patch_box)
                   mix_gt_boxes[mix_idx,patch,5:] = mix_onehot[patch]*mix_lambda + onehot_tgt[tgt_patch_idx]*(1-mix_lambda)#put it into mix_gt_boxes
            else:
               for tgt_patch_idx in range(tgt_num_boxes[0]):
                   patch = 0 if num_boxes[mix_idx]==1 else random.randint(0,num_boxes[mix_idx]-1)

                   tgt_patch_box = tgt_gt_boxes[0,tgt_patch_idx,:]
                   tgt_patch = tgt_im_data[0,:,tgt_patch_box[1].type(torch.int):tgt_patch_box[3].type(torch.int),tgt_patch_box[0].type(torch.int):tgt_patch_box[2].type(torch.int)].unsqueeze(0)
                   
                   resized_tgt_patch = F.interpolate(tgt_patch, size=((gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int)),(gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int)))).squeeze(0)
                   #print(im_data[mix_idx].size(),mix_gt_boxes[mix_idx,num_boxes[mix_idx]-1,:],patch)
                   #print(gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int),gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int))
                   #print(resized_tgt_patch.size())
                   #print(gt_boxes[mix_idx,patch,1].type(torch.int)-gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int)-gt_boxes[mix_idx,patch,2].type(torch.int))
                   #print(mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)].shape,(resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)).shape)
                   mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)] += resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)
                   
                   tgt_patch_box_list.append(tgt_patch_box)
                   
                   mix_gt_boxes[mix_idx,patch,5:] = mix_onehot[patch]*mix_lambda + onehot_tgt[tgt_patch_idx]*(1-mix_lambda)#put it into mix_gt_boxes
                   
            
            tgt_im_data_copy = tgt_im_data.detach().clone()
            tgt_box_x_min = min(tgt_gt_boxes[0,:tgt_num_boxes[0],0]).type(torch.int)
            tgt_box_x_max = max(tgt_gt_boxes[0,:tgt_num_boxes[0],2]).type(torch.int)
            tgt_box_y_min = min(tgt_gt_boxes[0,:tgt_num_boxes[0],1]).type(torch.int)
            tgt_box_y_max = max(tgt_gt_boxes[0,:tgt_num_boxes[0],3]).type(torch.int)
            #print(tgt_gt_boxes)
            #print(tgt_num_boxes[0],tgt_gt_boxes[0,:tgt_num_boxes[0],0],tgt_gt_boxes[0,:tgt_num_boxes[0],2])
            left_area = tgt_box_x_min*tgt_im_data.size(2)
            right_area = (tgt_im_data.size(3)-tgt_box_x_max)*tgt_im_data.size(2)
            up_area = tgt_box_y_min*tgt_im_data.size(3)
            down_area = (tgt_im_data.size(2)-tgt_box_y_max)*tgt_im_data.size(3)
            src_tgt_x_ratio = im_data.size(3)/tgt_im_data.size(3)
            src_tgt_y_ratio = im_data.size(2)/tgt_im_data.size(2)
            #print("tgt_box_x_min:",tgt_box_x_min,"tgt_box_x_max:",tgt_box_x_max,"tgt_box_y_min:",tgt_box_y_min,"tgt_box_y_max:",tgt_box_y_max)
            #print("left_area:",left_area,"right_area:",right_area,"up_area:",up_area,"down_area:",down_area)
            #print("im_data_w:",im_data.size(3),"tgt_im_data_w:",tgt_im_data.size(3),"src_tgt_x_ratio:",src_tgt_x_ratio)
            #print("im_data_h:",im_data.size(2),"tgt_im_data_h:",tgt_im_data.size(2),"src_tgt_y_ratio:",src_tgt_y_ratio)
            repeat_count=0
            x_flag=0
            if max([left_area,right_area,up_area,down_area])==left_area:
                x_flag=1
                tgt_back_x_min=0
                tgt_back_x_max=tgt_box_x_min
                tgt_back_y_min=0
                tgt_back_y_max=tgt_im_data.size(2)
                
                back_x_min=0
                back_x_max=((tgt_back_x_max-tgt_back_x_min).type(torch.float)*src_tgt_x_ratio).type(torch.int)
                back_y_min=0
                back_y_max=im_data.size(2)
                repeat_count = math.ceil((im_data.size(3)/back_x_max))
            elif max([left_area,right_area,up_area,down_area])==right_area:
                x_flag=1
                tgt_back_x_min=tgt_box_x_max
                tgt_back_x_max=tgt_im_data.size(3)
                tgt_back_y_min=0
                tgt_back_y_max=tgt_im_data.size(2)

                back_x_min=0
                back_x_max=((tgt_back_x_max-tgt_back_x_min).type(torch.float)*src_tgt_x_ratio).type(torch.int)
                #print(tgt_back_x_max-tgt_back_x_min,(tgt_back_x_max-tgt_back_x_min).type(torch.float)*src_tgt_x_ratio,back_x_max)
                back_y_min=0
                back_y_max=im_data.size(2)
                repeat_count = math.ceil(im_data.size(3)/back_x_max)
            elif max([left_area,right_area,up_area,down_area])==up_area:
                x_flag=0
                tgt_back_x_min=0
                tgt_back_x_max=tgt_im_data.size(3)
                tgt_back_y_min=0
                tgt_back_y_max=tgt_box_y_min
    
                back_x_min=0
                back_x_max=im_data.size(3)
                back_y_min=0
                back_y_max=((tgt_back_y_max-tgt_back_y_min).type(torch.float)*src_tgt_y_ratio).type(torch.int)
                repeat_count = math.ceil(im_data.size(2)/back_y_max)
            elif max([left_area,right_area,up_area,down_area])==down_area:
                x_flag=0
                tgt_back_x_min=0
                tgt_back_x_max=tgt_im_data.size(3)
                tgt_back_y_min=tgt_box_y_max
                tgt_back_y_max=tgt_im_data.size(2)
  
                back_x_min=0
                back_x_max=im_data.size(3)
                back_y_min=0
                back_y_max=((tgt_back_y_max-tgt_back_y_min).type(torch.float)*src_tgt_y_ratio).type(torch.int)
                repeat_count = math.ceil(im_data.size(2)/back_y_max)
            #print(tgt_back_x_min,tgt_back_x_max,tgt_back_y_min,tgt_back_y_max)
            #print(back_x_min,back_x_max,back_y_min,back_y_max)
            #print(repeat_count)
            for repeat_idx in range(repeat_count):
                if x_flag==1:
                    #print(back_x_min+back_x_max*repeat_idx,min(back_x_max+back_x_max*repeat_idx, im_data.size(3)),tgt_back_x_min,tgt_back_x_max,back_x_max)
                    if back_x_max+back_x_max*repeat_idx>im_data.size(3):
                        mix_img[mix_idx,:,:im_data.size(2),(back_x_min+back_x_max*repeat_idx):min(back_x_max+back_x_max*repeat_idx, im_data.size(3))] += F.interpolate(tgt_im_data_copy[:,:,:,tgt_back_x_min:tgt_back_x_max], size=(im_data.size(2), back_x_max))[:,:,:,:(back_x_max+back_x_max*repeat_idx-im_data.size(3))].squeeze(0).cpu().numpy()*(1-mix_lambda)
                    else:
                        mix_img[mix_idx,:,:im_data.size(2),(back_x_min+back_x_max*repeat_idx):min(back_x_max+back_x_max*repeat_idx, im_data.size(3))] += F.interpolate(tgt_im_data_copy[:,:,:,tgt_back_x_min:tgt_back_x_max], size=(im_data.size(2), back_x_max)).squeeze(0).cpu().numpy()*(1-mix_lambda)
                else:
                    if back_y_max+back_y_max*repeat_idx>im_data.size(2):
                        mix_img[mix_idx,:,back_y_min+back_y_max*repeat_idx:min(back_y_max+back_y_max*repeat_idx, im_data.size(2)),:im_data.size(3)] += F.interpolate(tgt_im_data_copy[:,:,tgt_back_y_min:tgt_back_y_max,:], size=(back_y_max, im_data.size(3)))[:,:,:(back_y_max+back_y_max*repeat_idx-im_data.size(2)),:].squeeze(0).cpu().numpy()*(1-mix_lambda)
                    else:
                        mix_img[mix_idx,:,back_y_min+back_y_max*repeat_idx:min(back_y_max+back_y_max*repeat_idx, im_data.size(2)),:im_data.size(3)] += F.interpolate(tgt_im_data_copy[:,:,tgt_back_y_min:tgt_back_y_max,:], size=(back_y_max, im_data.size(3))).squeeze(0).cpu().numpy()*(1-mix_lambda)
            #mix_img[mix_idx,:,:im_data.size(2),:im_data.size(3)] += F.interpolate(tgt_im_data_copy, size=(im_data.size(2), im_data.size(3))).squeeze(0).cpu().numpy()*(1-mix_lambda)
        mix_im_data = torch.from_numpy(mix_img).cuda()
        return mix_im_data, mix_im_info, mix_gt_boxes, mix_num_boxes

    def ourmixup(self, batch_size, im_data, im_info, gt_boxes, num_boxes, tgt_im_data,tgt_gt_boxes,tgt_num_boxes):
        mix_lambda=max(0,min(1,np.random.beta(0.2,0.2)))#random.uniform(0.7,1)#
        #mix_label_lambda = max(0,min(1,np.random.beta(0.2,0.2)))
        mix_height = im_data.size(2)
        mix_width = im_data.size(3)
        mix_img = np.zeros((batch_size, im_data.size(1), mix_height, mix_width), dtype=np.float32)
        mix_im_info = im_info.detach().clone()
        mix_gt_boxes = gt_boxes.detach().clone()
        mix_num_boxes = num_boxes.detach().clone()
        #print(mix_gt_boxes.size())#(3,50,5)add one hot space here
        mix_gt_boxes_onehot_append = torch.zeros([gt_boxes.size(0), gt_boxes.size(1),self.n_classes]).cuda()
        mix_gt_boxes = torch.cat((mix_gt_boxes,mix_gt_boxes_onehot_append),-1)
        #print(mix_gt_boxes.size())#(3,50,11)
        
        for mix_idx in range(batch_size):
            
            mix_im_info[mix_idx][2] = 1.
            
            onehot_im = self.onehot(mix_gt_boxes[mix_idx,:,4].long().view(-1,1),self.n_classes)
            mix_gt_boxes[mix_idx,:,5:] = onehot_im
            onehot_tgt = self.onehot(tgt_gt_boxes[0, :,4].long().view(-1,1),self.n_classes)
            #print(onehot_im.size(),onehot_tgt.size())#(50,6)
            mix_onehot = onehot_im.detach().clone()
            mix_img[mix_idx,:,:im_data.size(2),:im_data.size(3)] = im_data[mix_idx].detach().clone().cpu().numpy()*mix_lambda
            tgt_patch_box_list = []
            if tgt_num_boxes[0]>num_boxes[mix_idx]: # past random tgt patch to im box one by one             
                for patch in range(num_boxes[mix_idx]):
                   tgt_patch_idx = random.randint(0,tgt_num_boxes[0]-1)
                   tgt_patch_box = tgt_gt_boxes[0,tgt_patch_idx,:]
                   #print("tgt+gt_box:",tgt_patch_box,tgt_im_data.shape)
                   tgt_patch = tgt_im_data[0,:,tgt_patch_box[1].type(torch.int):tgt_patch_box[3].type(torch.int),tgt_patch_box[0].type(torch.int):tgt_patch_box[2].type(torch.int)].unsqueeze(0)
                   #print(tgt_patch.size())
                   resized_tgt_patch = F.interpolate(tgt_patch, size=((gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int)),(gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int)))).squeeze(0)
                   #print(im_data[mix_idx].size(),mix_gt_boxes[mix_idx,num_boxes[mix_idx]-1,:],patch)
                   #print(gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int),gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int))
                   #print(resized_tgt_patch.size())
                   #print(gt_boxes[mix_idx,patch,1].type(torch.int)-gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int)-gt_boxes[mix_idx,patch,2].type(torch.int))
                   #print(mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)].shape,(resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)).shape)
                   mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)] += resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)
                   tgt_patch_box_list.append(tgt_patch_box)
                   mix_gt_boxes[mix_idx,patch,5:] = mix_onehot[patch]*mix_lambda + onehot_tgt[tgt_patch_idx]*(1-mix_lambda)#put it into mix_gt_boxes
            else:
               for tgt_patch_idx in range(tgt_num_boxes[0]):
                   patch = 0 if num_boxes[mix_idx]==1 else random.randint(0,num_boxes[mix_idx]-1)
                   tgt_patch_box = tgt_gt_boxes[0,tgt_patch_idx,:]
                   #print("tgt+gt_box:",tgt_patch_box,tgt_im_data.shape)
                   tgt_patch = tgt_im_data[0,:,tgt_patch_box[1].type(torch.int):tgt_patch_box[3].type(torch.int),tgt_patch_box[0].type(torch.int):tgt_patch_box[2].type(torch.int)].unsqueeze(0)
                   #print(tgt_patch.size())
                   resized_tgt_patch = F.interpolate(tgt_patch, size=((gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int)),(gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int)))).squeeze(0)
                   #print(im_data[mix_idx].size(),mix_gt_boxes[mix_idx,num_boxes[mix_idx]-1,:],patch)
                   #print(gt_boxes[mix_idx,patch,3].type(torch.int)-gt_boxes[mix_idx,patch,1].type(torch.int),gt_boxes[mix_idx,patch,2].type(torch.int)-gt_boxes[mix_idx,patch,0].type(torch.int))
                   #print(resized_tgt_patch.size())
                   #print(gt_boxes[mix_idx,patch,1].type(torch.int)-gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int)-gt_boxes[mix_idx,patch,2].type(torch.int))
                   #print(mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)].shape,(resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)).shape)
                   mix_img[mix_idx,:,gt_boxes[mix_idx,patch,1].type(torch.int):gt_boxes[mix_idx,patch,3].type(torch.int),gt_boxes[mix_idx,patch,0].type(torch.int):gt_boxes[mix_idx,patch,2].type(torch.int)] += resized_tgt_patch.detach().clone().cpu().numpy()*(1-mix_lambda)
                   
                   tgt_patch_box_list.append(tgt_patch_box)
                   
                   mix_gt_boxes[mix_idx,patch,5:] = mix_onehot[patch]*mix_lambda + onehot_tgt[tgt_patch_idx]*(1-mix_lambda)#put it into mix_gt_boxes
                   
            
            tgt_im_data_copy = tgt_im_data.detach().clone()
            for blank_tgt_idx in range(tgt_num_boxes[0]):
                tgt_im_data_copy[0,:,tgt_gt_boxes[0,blank_tgt_idx,1].type(torch.int):tgt_gt_boxes[0,blank_tgt_idx,3].type(torch.int),tgt_gt_boxes[0,blank_tgt_idx,0].type(torch.int):tgt_gt_boxes[0,blank_tgt_idx,2].type(torch.int)] = 0
            mix_img[mix_idx,:,:im_data.size(2),:im_data.size(3)] += F.interpolate(tgt_im_data_copy, size=(im_data.size(2), im_data.size(3))).squeeze(0).cpu().numpy()*(1-mix_lambda)
        mix_im_data = torch.from_numpy(mix_img).cuda()
        return mix_im_data, mix_im_info, mix_gt_boxes, mix_num_boxes

    def ourmixup_s_t(self, batch_size, im_data, im_info, gt_boxes, num_boxes, tgt_im_data,tgt_im_info,tgt_gt_boxes,tgt_num_boxes):
        mix_lambda=max(0,min(1,np.random.beta(0.2,0.2)))#random.uniform(0.7,1)#
        mix_height = tgt_im_data.size(2)
        mix_width = tgt_im_data.size(3)
        mix_img = np.zeros((batch_size, im_data.size(1), mix_height, mix_width), dtype=np.float32)
        mix_im_info = tgt_im_info.detach().clone()
        print(mix_im_info.size(),mix_im_info,mix_height, mix_width)
        mix_gt_boxes = tgt_gt_boxes.detach().clone()
        mix_num_boxes = tgt_num_boxes.detach().clone()
        print(mix_num_boxes)
        #print(mix_gt_boxes.size())#(3,50,5)add one hot space here
        mix_gt_boxes_onehot_append = torch.zeros([tgt_gt_boxes.size(0), tgt_gt_boxes.size(1),self.n_classes]).cuda()
        #triple above
        mix_gt_boxes = torch.cat((mix_gt_boxes,mix_gt_boxes_onehot_append),-1)
        #print(mix_gt_boxes.size())#(3,50,13)
        
        for mix_idx in range(batch_size):
            
            mix_im_info[mix_idx][2] = 1.
            
            onehot_tgt = self.onehot(mix_gt_boxes[mix_idx,:,4].long().view(-1,1),self.n_classes)
            mix_gt_boxes[mix_idx,:,5:] = onehot_tgt
            onehot_im = self.onehot(gt_boxes[mix_idx, :,4].long().view(-1,1),self.n_classes)
            #print(onehot_im.size(),onehot_tgt.size())#(50,8)
            mix_onehot = onehot_tgt.detach().clone()
            mix_img[mix_idx,:,:tgt_im_data.size(2),:tgt_im_data.size(3)] = tgt_im_data[0].detach().clone().cpu().numpy()*mix_lambda
            patch_box_list = []
            for patch in range(tgt_num_boxes[0]):
               patch_idx = 0
               patch_box = gt_boxes[0,patch_idx,:]
               im_patch = im_data[mix_idx,:,patch_box[1].type(torch.int):patch_box[3].type(torch.int),patch_box[0].type(torch.int):patch_box[2].type(torch.int)].unsqueeze(0)
               resized_im_patch = F.interpolate(im_patch, size=((tgt_gt_boxes[0,patch,3].type(torch.int)-tgt_gt_boxes[0,patch,1].type(torch.int)),(tgt_gt_boxes[0,patch,2].type(torch.int)-tgt_gt_boxes[0,patch,0].type(torch.int)))).squeeze(0)
               mix_img[mix_idx,:,tgt_gt_boxes[0,patch,1].type(torch.int):tgt_gt_boxes[0,patch,3].type(torch.int),tgt_gt_boxes[0,patch,0].type(torch.int):tgt_gt_boxes[0,patch,2].type(torch.int)] += resized_im_patch.detach().clone().cpu().numpy()*(1-mix_lambda)
               patch_box_list.append(patch_box)
               mix_gt_boxes[mix_idx,patch,5:] = mix_onehot[patch]*mix_lambda + onehot_im[patch_idx]*(1-mix_lambda)#put it into mix_gt_boxes
            
            im_data_copy = im_data.detach().clone()
            im_data_copy[mix_idx,:,gt_boxes[mix_idx,0,1].type(torch.int):gt_boxes[mix_idx,0,3].type(torch.int),gt_boxes[mix_idx,0,0].type(torch.int):gt_boxes[mix_idx,0,2].type(torch.int)] = 0
            mix_img[mix_idx,:,:tgt_im_data.size(2),:tgt_im_data.size(3)] += F.interpolate(im_data_copy[mix_idx], size=(tgt_im_data.size(2), tgt_im_data.size(3))).squeeze(0).cpu().numpy()*(1-mix_lambda)
        mix_im_data = torch.from_numpy(mix_img).cuda()
        return mix_im_data, mix_im_info, mix_gt_boxes, mix_num_boxes

    def onehot(self, labels, label_num):
        return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(1,labels.view(-1,1),1)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop, itersnum,newepoch=1):

        #print(need_backprop)
        #print(tgt_need_backprop)
        #assert need_backprop.detach()[0]==1 and tgt_need_backprop.detach()==0

        batch_size = im_data.size(0)
        #print("source size", tgt_im_data.size())
        im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop=need_backprop.data
        
        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)



        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data #rois_label(3,256), rois_target(3,256,4)
            

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        
        #
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            #print("source roi labels",rois_label.size())
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            #onehot_rois_label = self.onehot(rois_label, self.n_classes)
            #print(onehot_rois_label.size())
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            #RCNN_loss_cls = -torch.mean(torch.sum(F.log_softmax(cls_score,dim=1)*onehot_rois_label, dim=1))

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        """ =================== for mixup =========================="""
        
        
        mix_im_data, mix_im_info, mix_gt_boxes, mix_num_boxes = self.ourmixup(batch_size, im_data, im_info, gt_boxes, num_boxes, tgt_im_data,tgt_gt_boxes,tgt_num_boxes)
        #print("gt after mixup",mix_gt_boxes[:,:,4:], torch.unique(mix_gt_boxes[:,:,4]), torch.unique(tgt_gt_boxes[:,:,4]))
        #print(num_boxes)

        # feed image data to base model to obtain base feature map
        mix_base_feat = self.RCNN_base(mix_im_data)



        # feed base feature map tp RPN to obtain rois
        #print(mix_gt_boxes.size())
        #print(mix_gt_boxes[:,:,:4].size())
        mix_rois, mix_rpn_loss_cls, mix_rpn_loss_bbox = self.RCNN_rpn(mix_base_feat, mix_im_info, mix_gt_boxes[:,:,:5], mix_num_boxes)
        #print(torch.unique(mix_rois[:,:,0]))#0,1,..batch_size-1

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            mix_roi_data = self.ourmixup_RCNN_proposal_target(mix_rois, mix_gt_boxes, mix_num_boxes)
            mix_rois, mix_rois_label, mix_rois_target, mix_rois_inside_ws, mix_rois_outside_ws = mix_roi_data
            #print(mix_rois_label.size())#(3,50,7)
            #print(torch.unique(mix_rois_label[:,:,0]))
            #print(torch.unique(gt_boxes[:,:,4]))
            

            mix_rois_label = Variable(mix_rois_label.view(-1,1+self.n_classes))
            mix_rois_target = Variable(mix_rois_target.view(-1, mix_rois_target.size(2)))
            mix_rois_inside_ws = Variable(mix_rois_inside_ws.view(-1, mix_rois_inside_ws.size(2)))
            mix_rois_outside_ws = Variable(mix_rois_outside_ws.view(-1, mix_rois_outside_ws.size(2)))
        else:
            mix_rois_label = None
            mix_rois_target = None
            mix_rois_inside_ws = None
            mix_rois_outside_ws = None
            mix_rpn_loss_cls = 0
            mix_rpn_loss_bbox = 0

        mix_rois = Variable(mix_rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            mix_pooled_feat = self.RCNN_roi_align(mix_base_feat, mix_rois[:,:,:5].view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            mix_pooled_feat = self.RCNN_roi_pool(mix_base_feat, mix_rois[:,:,:5].view(-1,5))

        
        # feed pooled features to top model
        mix_pooled_feat = self._head_to_tail(mix_pooled_feat)
        
        #
        # compute bbox offset
        mix_bbox_pred = self.RCNN_bbox_pred(mix_pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            mix_bbox_pred_view = mix_bbox_pred.view(mix_bbox_pred.size(0), int(mix_bbox_pred.size(1) / 4), 4)
            mix_bbox_pred_select = torch.gather(mix_bbox_pred_view, 1, mix_rois_label[:,0].long().view(mix_rois_label.size(0), 1, 1).expand(mix_rois_label.size(0), 1, 4))
            mix_bbox_pred = mix_bbox_pred_select.squeeze(1)

        # compute object classification probability
        mix_cls_score = self.RCNN_cls_score(mix_pooled_feat)
        mix_cls_prob = F.softmax(mix_cls_score, 1)

        mix_RCNN_loss_cls = 0
        mix_RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            #onehot_rois_label = self.onehot(rois_label, self.n_classes)
            #print(onehot_rois_label.size())
            #print(mix_rois_label,mix_rois_label[:,1:].dtype)
            mix_RCNN_loss_cls = -torch.mean(torch.sum(F.log_softmax(cls_score,dim=1)*mix_rois_label[:,1:], dim=1))
            #mix_RCNN_loss_cls = F.cross_entropy(mix_cls_score, mix_rois_label)

            # bounding box regression L1 loss
            mix_RCNN_loss_bbox = _smooth_l1_loss(mix_bbox_pred, mix_rois_target, mix_rois_inside_ws, mix_rois_outside_ws)


        mix_cls_prob = mix_cls_prob.view(batch_size, mix_rois.size(1), -1)
        mix_bbox_pred = mix_bbox_pred.view(batch_size, mix_rois.size(1), -1)
        

        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        #print("target size", tgt_im_data.size())
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data
        tgt_need_backprop = tgt_need_backprop.data

        # feed image data to base model to obtain base feature map
        tgt_base_feat = self.RCNN_base(tgt_im_data)
        #print("source base feat size", base_feat.size())
        #print("target base feat size", tgt_base_feat.size())


        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = \
            self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)
        #print(tgt_rois)
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            tgt_roi_data = self.RCNN_proposal_target(tgt_rois, tgt_gt_boxes,tgt_num_boxes)
            tgt_rois, tgt_rois_label, tgt_rois_target, tgt_rois_inside_ws, tgt_rois_outside_ws = tgt_roi_data

            tgt_rois_label = Variable(tgt_rois_label.view(-1).long())
            tgt_rois_target = Variable(tgt_rois_target.view(-1, tgt_rois_target.size(2)))
            tgt_rois_inside_ws = Variable(tgt_rois_inside_ws.view(-1, tgt_rois_inside_ws.size(2)))
            tgt_rois_outside_ws = Variable(tgt_rois_outside_ws.view(-1, tgt_rois_outside_ws.size(2)))
        else:
            tgt_rois_label = None
            tgt_rois_target = None
            tgt_rois_inside_ws = None
            tgt_rois_outside_ws = None
            tgt_rpn_loss_cls = 0
            tgt_rpn_loss_bbox = 0

        tgt_rois = Variable(tgt_rois)
        # do roi pooling based on predicted rois
        #print("tgt)base_feat size", tgt_base_feat.size())
        #print(tgt_rois.view(-1,5))
        if cfg.POOLING_MODE == 'align':
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))

        #print("tgt_pooled_feat size",tgt_pooled_feat.size())
        # feed pooled features to top model
        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)
        #print("tgt_pooled_feat size",tgt_pooled_feat.size())
        #
        # compute bbox offset
        tgt_bbox_pred = self.RCNN_bbox_pred(tgt_pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            tgt_bbox_pred_view = tgt_bbox_pred.view(tgt_bbox_pred.size(0), int(tgt_bbox_pred.size(1) / 4), 4)
            tgt_bbox_pred_select = torch.gather(tgt_bbox_pred_view, 1, tgt_rois_label.view(tgt_rois_label.size(0), 1, 1).expand(tgt_rois_label.size(0), 1, 4))
            tgt_bbox_pred = tgt_bbox_pred_select.squeeze(1)

        # compute object classification probability
        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat)
        tgt_cls_prob = F.softmax(tgt_cls_score, 1)

        tgt_RCNN_loss_cls = 0
        tgt_RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            tgt_RCNN_loss_cls = F.cross_entropy(tgt_cls_score, tgt_rois_label)

            # bounding box regression L1 loss
            tgt_RCNN_loss_bbox = _smooth_l1_loss(tgt_bbox_pred, tgt_rois_target, tgt_rois_inside_ws, tgt_rois_outside_ws)


        tgt_cls_prob = tgt_cls_prob.view(tgt_batch_size, tgt_rois.size(1), -1)
        tgt_bbox_pred = tgt_bbox_pred.view(tgt_batch_size, tgt_rois.size(1), -1)


        
        """  DA loss   """

        # DA LOSS
        DA_img_loss_cls = 0
        DA_ins_loss_cls = 0

        tgt_DA_img_loss_cls = 0
        tgt_DA_ins_loss_cls = 0

        
        
        s_s_base_feat_list = []
        s_t_base_feat_list = []
        count = 0
        #print(base_feat.size())
        for _, (scales,imageDA_batch_size) in enumerate(zip([64,32,16],[50,50,50])):
            #print(scales, imageDA_batch_size/2)
            anchors = generate_anchors(base_size=16, ratios=np.array([0.5,1,2]), scales=np.array([scales]))
            #print(anchors)
            #print("w",anchors[:,2]-anchors[:,0])
            #print("h", anchors[:,3]-anchors[:,1])
            for imageDA_idx in range(round(imageDA_batch_size/2)):
                
                #print(grid)
                for pair_idx in range(tgt_batch_size):
                    grid = anchors[imageDA_idx%len(anchors)]
                    #if batch_size!=3:
                    #    print("batch_size!=3")
                    #if pair_idx!=0:
                    #    print("pair_idx error")
                    offsetx = random.randint(0, grid[2]-grid[0])
                    offsety = random.randint(0, grid[3]-grid[1])
                    grid[0] = np.clip(grid[0] + offsetx, 0, im_data.size(3))
                    grid[2] = np.clip(grid[2] + offsetx, 0, im_data.size(3))
                    grid[1] = np.clip(grid[1] + offsety, 0, im_data.size(2))
                    grid[3] = np.clip(grid[3] + offsety, 0, im_data.size(2))
                    #print(grid)
                    grid = torch.tensor(grid).cuda()
                    #print(grid)
                    test_num=3
                    base_grid = grid.new(test_num,1,5).zero_()
                    for s_b_idx in range(test_num):
                        base_grid[s_b_idx,:,0] = s_b_idx
                        base_grid[s_b_idx,:,1:] = grid
                    base_grid = base_grid.to(torch.float32)
                    #print(base_grid.view(-1,5))
                    #print(im_data.size())
                    tgt_base_grid = grid.new(tgt_batch_size,1,5).zero_()
                    for t_b_idx in range(tgt_batch_size):
                        tgt_base_grid[t_b_idx,:,0] = t_b_idx
                        tgt_base_grid[t_b_idx,:,1:] = grid
                    tgt_base_grid = tgt_base_grid.to(torch.float32)
                    
                    #base_feat_sp_1 = self.split_pool(base_feat[pair_idx:(pair_idx+1)],base_grid.view(-1,5))
                    #base_feat_sp_2 = self.split_pool(base_feat[(pair_idx+1):(pair_idx+2)],base_grid.view(-1,5))
                    #base_feat_sp_3 = self.split_pool(base_feat[(pair_idx+2):(pair_idx+3)],base_grid.view(-1,5))
                    
                    #print(base_feat_sp.size())
                    #print(base_feat[(pair_idx*batch_size):(pair_idx*batch_size+batch_size)].size())
                    base_feat_sp = self.split_pool(base_feat[(pair_idx*batch_size):(pair_idx*batch_size+batch_size)], base_grid.view(-1,5))
                    tgt_base_feat_sp = self.split_pool(tgt_base_feat[pair_idx:(pair_idx+tgt_batch_size)], tgt_base_grid.view(-1,5))
                    
                    #print(tgt_base_grid.size())
                    #print(base_feat_sp.size())
                    #print(base_feat_sp[0].size())
                    #print(base_feat_sp[1])
                    #print(tgt_base_feat_sp.size())
                    
                    s_s_base_feat = torch.cat([base_feat_sp[0], base_feat_sp[1]],0)
                    #print(s_s_base_feat.size())
                    #s_s_base_feat = s_s_base_feat.view(1, s_s_base_feat.size(0), s_s_base_feat.size(1), s_s_base_feat.size(2))
                    #s_s_base_feat_list.append(s_s_base_feat)
                    
                    if count==0:
                        s_s_base_feat_pair = s_s_base_feat
                    elif count==1:
                        s_s_base_feat_pair = torch.stack([s_s_base_feat_pair, s_s_base_feat], dim=0)
                    else:
                        #print(s_s_base_feat.unsqueeze(0).size())
                        s_s_base_feat_pair = torch.cat((s_s_base_feat_pair,s_s_base_feat.unsqueeze(0)), dim=0)
                    #print(s_s_base_feat_pair.size())
                    #print(base_feat[0].size(), s_s_base_feat.size())
                    s_t_base_feat = torch.cat([base_feat_sp[2], tgt_base_feat_sp[0]],0)
                    #s_t_base_feat = s_t_base_feat.view(1, s_t_base_feat.size(0), s_t_base_feat.size(1), s_t_base_feat.size(2))
                    #s_t_base_feat_list.append(s_t_base_feat)
                    if count==0:
                        s_t_base_feat_pair = s_t_base_feat
                        count=1
                    elif count==1:
                        s_t_base_feat_pair = torch.stack([s_t_base_feat_pair, s_t_base_feat], dim=0)
                        count=2
                    else:
                        s_t_base_feat_pair = torch.cat((s_t_base_feat_pair, s_t_base_feat.unsqueeze(0)),dim=0)
                    
                    #print(s_s_base_feat_pair.size())
        #s_s_base_feat_pair = torch.stack(s_s_base_feat_list, dim=0)
        #s_t_base_feat_pair = torch.stack(s_t_base_feat_list, dim=0)
        #print(s_s_base_feat_pair.size())
        #print(s_t_base_feat_pair.size())
        
        class_in_source = torch.unique(rois_label.cpu()).cuda()
        class_in_target = torch.unique(tgt_rois_label.cpu()).cuda()
        class_to_instanceDA = [val for val in class_in_source if val in class_in_target]
        class_rois_label_index_dict = {key:(rois_label==key).nonzero().squeeze(1) for key in class_to_instanceDA}
        class_tgt_rois_label_index_dict = {key:(tgt_rois_label==key).nonzero().squeeze(1) for key in class_to_instanceDA}
        #print(torch.unique(rois_label.cpu()))
        #for l in class_in_source:
        #    print(len([val for val in rois_label if val==l]))
        #print(torch.unique(tgt_rois_label.cpu()))
        #for l in class_in_target:
        #    print(len([val for val in tgt_rois_label if val==l]))
        #print(class_to_instanceDA)
        #print(len(class_rois_label_index_dict[class_to_instanceDA[0]]))
        
        s_s_pooled_feat_list = []
        s_s_instance_label = []
        s_t_pooled_feat_list = []
        s_t_instance_label = []
        #print(pooled_feat.size())
        #print(rois_label.size())
        for class_i in class_to_instanceDA:
            source_class_index = class_rois_label_index_dict[class_i]
            target_class_index = class_tgt_rois_label_index_dict[class_i]
            for pair_idx in range(len(target_class_index)):
                if len(source_class_index)>=3:
                    random_3index = torch.randint(len(source_class_index), (3,)).long()
                elif len(source_class_index)==2:
                    random_3index = torch.tensor([0,1,0]).long()
                else:
                    random_3index = torch.tensor([0,0,0]).long()
                #print(random_3index)
                selected_index = source_class_index[random_3index]
                #print(selected_index)
                selected_source_pooled_feat = pooled_feat[selected_index]
                #print(selected_source_pooled_feat.size())
                
                #print(selected_source_pooled_feat)
                s_s_ins_feat = torch.cat([selected_source_pooled_feat[0], selected_source_pooled_feat[1]],0)
                s_s_pooled_feat_list.append(s_s_ins_feat)
                s_s_instance_label.append(1)#1#class_i
                #print(s_s_ins_feat.size())
                #print(selected_source_pooled_feat[2].size())
                #print(tgt_pooled_feat[target_class_index[pair_idx]].size())
                s_t_ins_feat = torch.cat([selected_source_pooled_feat[2], tgt_pooled_feat[target_class_index[pair_idx]]],0)
                s_t_pooled_feat_list.append(s_t_ins_feat)
                s_t_instance_label.append(0)#class_i+self.n_classes#0
                #print(s_t_base_feat.size())
        s_s_pooled_feat_pair = torch.stack(s_s_pooled_feat_list, dim=0)
        s_t_pooled_feat_pair = torch.stack(s_t_pooled_feat_list, dim=0)
        #print(s_s_pooled_feat_pair.size()) 
        
        
        base_score = self.RCNN_imageDA(s_s_base_feat_pair)
        #base_score, base_label = self.RCNN_imageDA(base_feat, need_backprop)
        # Image DA

        #print(base_score.size(), max(base_score), min(base_score))
        #print(tgt_need_backprop+1)
        #print(im_data)
        image_label = (tgt_need_backprop+1).expand(len(base_score))
        image_loss = nn.BCELoss()
        #print(base_score,base_score.size())
        #print(image_label.size())
        DA_img_loss_cls = image_loss(base_score, image_label)
        
        
        instance_sigmoid = self.RCNN_instanceDA(s_s_pooled_feat_pair)
        #print(instance_sigmoid.size())
        DA_ins_loss_cls = F.cross_entropy(instance_sigmoid, torch.tensor(s_s_instance_label).cuda())
        
        #consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
        #consistency_prob = F.softmax(instance_sigmoid, dim=1)[:,1,:,:]
        #consistency_prob=torch.mean(consistency_prob)
        #consistency_prob=consistency_prob.repeat(base_score.size())

        #DA_cst_loss=self.consistency_loss(base_score,consistency_prob.detach())

        """  ************** taget loss ****************  """
         
        tgt_base_score = self.RCNN_imageDA(s_t_base_feat_pair)
        tgt_image_label = tgt_need_backprop.expand(len(tgt_base_score))
        # Image DA
        tgt_image_loss = nn.BCELoss()
        tgt_DA_img_loss_cls = tgt_image_loss(tgt_base_score, tgt_image_label)
        
        
        
        tgt_instance_sigmoid = self.RCNN_instanceDA(s_t_pooled_feat_pair)
        tgt_DA_ins_loss_cls = F.cross_entropy(tgt_instance_sigmoid, torch.tensor(s_t_instance_label).cuda())
        

        #tgt_consistency_prob = F.softmax(tgt_instance_sigmoid, dim=1)[:, 0, :, :]
        #tgt_consistency_prob = torch.mean(tgt_consistency_prob)
        #tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_base_score.size())

        #tgt_DA_cst_loss = self.consistency_loss(tgt_base_score, tgt_consistency_prob.detach())
        

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
    mix_rois, mix_cls_prob, mix_bbox_pred, mix_rpn_loss_cls, mix_rpn_loss_bbox, mix_RCNN_loss_cls, mix_RCNN_loss_bbox, mix_rois_label,\
    tgt_rois, tgt_cls_prob, tgt_bbox_pred, tgt_rpn_loss_cls, tgt_rpn_loss_bbox, tgt_RCNN_loss_cls, tgt_RCNN_loss_bbox,\
    tgt_rois_label,DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_img_loss_cls,tgt_DA_ins_loss_cls#,DA_cst_loss,tgt_DA_cst_loss


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
