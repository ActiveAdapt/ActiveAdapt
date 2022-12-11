# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
#from Model_FasterRCNN import LemurModel, LemurSpecies
from model.utils.net_utils import save_net, load_net, vis_detections,save_checkpoint
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.utils.config import cfg,cfg_from_list,get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from judgecluster import judgecluster
from judgesimilarity import judgesimilarity
from unity_MIR import MIR
#from unity_strawmanMIR import MIR
from unity_finetune import finetune
from torch.autograd import Variable
import pandas as pd
import copy
from unity_evaluate_model import evaluate_model

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

from model.utils.blob import im_list_to_blob

import torch
import io, json
import os.path
from pathlib import Path
import cv2 as cv
import numpy as np
import base64
import matplotlib.pyplot as plt
from time import sleep, time
import datetime
from PIL import Image as im
import threading
from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# 2. Create app and model objects
app = FastAPI()

set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
cfg_from_list(set_cfgs)
cfg.CUDA = True

#model = LemurModel()
classes=('__background__',  # always index 0
                         'black-and-white-ruffed-lemur', 'blue-eyed-black-lemur', 'coquerels-sifaka',
                         'red-ruffed-lemur', 'ring-tailed-lemur')
model = vgg16(classes, pretrained=False, class_agnostic=False)
model.create_architecture()
#initial for lemur guidance
#checkpoint = torch.load("/data/ztc/adaptation/Experiment/model/vgg16/cleanlemur/finetune_cleanlemur_ours_epoch12_.pth")
#initial for source training
#checkpoint = torch.load("/data/ztc/adaptation/Experiment/da_model/vgg16/cleanlemur/train_cleanlemur_source.pth")
#initial for ft
checkpoint = torch.load("/data/ztc/adaptation/Experiment/model/vgg16/cleanlemur/strawman_initial.pth")
model.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in model.state_dict()})
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
model.cuda()
print("Load model successful")

initial_model=copy.deepcopy(model)
'''
checkpoint = torch.load("/data/ztc/adaptation/Experiment/model/vgg16/cleanlemur/guidance_toysystem_low_loop6.pth")
model.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in model.state_dict()})
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
model.cuda()
'''
empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
max_per_image=1
num_classes=6
class_idx_name={1:'BLACK-AND-WHITE-RUFFED-LEMUR',2:'BLUE-EYED-BLACK-LEMUR',3:'COQUERELS-SIFAKA',4:'RED-RUFFED-LEMUR',5:'RING-TAILED-LEMUR'}
idx_class_name={'BLACK-AND-WHITE-RUFFED-LEMUR':1,'BLUE-EYED-BLACK-LEMUR':2,'COQUERELS-SIFAKA':3,'RED-RUFFED-LEMUR':4,'RING-TAILED-LEMUR':5}

round_num=20#before is 20
retrive_num=50#before is 50
round_cnt=0
incom_name_list=[]
name_prediction_dict={}
score_set=[]
label_per=0.5
width=720#1440
height=1480#2960

guidance_threshold=0.1#0.1
back_div_weight=0.7#
ins_div_weight=0.3#
div_weight=0.9#0.9
uncertain_weight=0.1#0.1

method='low'
loop_count=1

center_list=None
pca=None
normlized_centers=None
x_min=None
x_max=None
ins_center_list=None
ins_pca=None
ins_normlized_centers=None
ins_x_min=None
ins_x_max=None

pre_im=None
training_t_set_name="training_set_t_CL_"+method
old_targetdata_name="old_targetdata_set_CL_"+method
old_data_name="old_data_set_CL_"+method


training_t_set=open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+training_t_set_name+"0.txt","w")
with open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/train_t_10fs.txt","r") as f:
  for line in f.readlines():
      training_t_set.write(line)
training_t_set.close()


old_targetdata_set=open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+old_targetdata_name+"0.txt","w")
with open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/train_t_10fs.txt","r") as f:
  for line in f.readlines():
    old_targetdata_set.write(line)
old_targetdata_set.close()


old_data_set=open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+old_data_name+"0.txt","w")
with open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/train_t_10fs.txt","r") as f:
  for line in f.readlines():
    old_data_set.write(line)
with open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/train_s.txt","r") as f:
  for line in f.readlines():
    old_data_set.write(line)
old_data_set.close()

def updatecluster(loop_count):
    global center_list
    global pca
    global normlized_centers
    global x_min
    global x_max
    global ins_center_list
    global ins_pca
    global ins_normlized_centers
    global ins_x_min
    global ins_x_max
    t1 = time()
    ################## center of gray scale of old target data
    print('loading lemur dataset.........')
    cfg.TRAIN.USE_FLIPPED = False
    old_t_imdb_name = "cleanlemur_2007_old_targetdata_set_CL_"+method+str(loop_count-1)#old_data_set_CL_high12"
   
    old_t_imdb, old_t_roidb, old_t_ratio_list, old_t_ratio_index = combined_roidb(old_t_imdb_name)
    old_t_train_size = len(old_t_roidb)   # add flipped         image_index*2
    old_t_num_images = len(old_t_imdb.image_index)
    print('{:d} roidb entries'.format(len(old_t_roidb)))

    old_t_dataset = roibatchLoader(old_t_roidb, old_t_ratio_list, old_t_ratio_index, 1, \
                           old_t_imdb.num_classes, training=False)

    old_t_dataloader = torch.utils.data.DataLoader(old_t_dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    # initilize the tensor holder here.
    old_t_im_data = torch.FloatTensor(1)
    old_t_im_data = old_t_im_data.cuda()
    old_t_num_boxes = torch.LongTensor(1)
    old_t_gt_boxes = torch.FloatTensor(1)
    old_t_num_boxes = old_t_num_boxes.cuda()
    old_t_gt_boxes = old_t_gt_boxes.cuda()

    # make variable
    old_t_im_data = Variable(old_t_im_data)
    old_t_gt_boxes = Variable(old_t_gt_boxes)
    old_t_num_boxes = Variable(old_t_num_boxes)

    old_t_data_iter = iter(old_t_dataloader)
    vector_list=np.zeros((old_t_num_images,360000))#grey scale 360000#flatten base 1420800#resize base 1280000 #head 4096
    instance_vector_list=None
    insNoneFlag=True
    old_t_im_name_list=[]
    for i in range(old_t_num_images):
      old_t_data = next(old_t_data_iter)
      old_t_im_name_list.append(old_t_data[-1][0])
    
      with torch.no_grad():
        old_t_im_data.resize_(old_t_data[0].size()).copy_(old_t_data[0])
        old_t_gt_boxes.resize_(old_t_data[2].size()).copy_(old_t_data[2])
        old_t_num_boxes.resize_(old_t_data[3].size()).copy_(old_t_data[3])
      img_gray=cv.cvtColor(old_t_im_data[0].detach().cpu().numpy().transpose(1,2,0),cv.COLOR_RGB2GRAY)
      img=cv.resize(img_gray,(600,600))
      img=img.reshape(1,360000)  
      vector=img
      vector_list[i]=vector

      ins_num=old_t_num_boxes[0].detach().cpu()
      
      for ins_idx in range(ins_num):
        ins_gt_box=old_t_gt_boxes[0][ins_idx,:].detach().cpu().numpy().astype(int)
        ins_x_min=ins_gt_box[0]
        ins_x_max=ins_gt_box[2]
        ins_y_min=ins_gt_box[1]
        ins_y_max=ins_gt_box[3]
        ins=img_gray[ins_y_min:ins_y_max,ins_x_min:ins_x_max]
        resized_ins=cv.resize(ins,(600,600))
        resized_ins=resized_ins.reshape(1,360000)  
        ins_vector=resized_ins
        if insNoneFlag==True:
          instance_vector_list=ins_vector
          insNoneFlag=False
        else:
          instance_vector_list=np.concatenate((instance_vector_list,ins_vector),axis=0)


    pca=PCA(n_components=3)
    pca_model=pca.fit(vector_list)
    pca_scale=pca.transform(vector_list)
    print(pca_scale.shape)
    pca_df_scale=pd.DataFrame(pca_scale,columns=['pc1','pc2','pc3'])
    
    kmeans_pca_scale = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)
    #print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
    center_list = kmeans_pca_scale.cluster_centers_
    print(center_list)
    x_min, x_max = np.min(pca_scale, 0), np.max(pca_scale, 0)
    normlized_centers = (center_list - x_min) / (x_max - x_min)

    ins_pca=PCA(n_components=3)
    ins_pca_model=ins_pca.fit(instance_vector_list)
    ins_pca_scale=ins_pca.transform(instance_vector_list)
    print(ins_pca_scale.shape)
    ins_pca_df_scale=pd.DataFrame(ins_pca_scale,columns=['pc1','pc2','pc3'])
    
    ins_kmeans_pca_scale = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(ins_pca_df_scale)
    #print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(ins_pca_df_scale, ins_kmeans_pca_scale.labels_, metric='euclidean')))
    ins_center_list = ins_kmeans_pca_scale.cluster_centers_
    print(ins_center_list)
    ins_x_min, ins_x_max = np.min(ins_pca_scale, 0), np.max(ins_pca_scale, 0)
    ins_normlized_centers = (ins_center_list - ins_x_min) / (ins_x_max - ins_x_min)



    ##################
    t2 = time()
    elapsed = t2 - t1
    print('Amount of time taken to update cluster: ' + str(elapsed) + '\n')
    print("!!!!!!!!!!!!!!!!Finish update cluster"+str(loop_count))

updatecluster(loop_count)

class ImgPathData(BaseModel):
    rgb_base64: str
    currentTime: str

class LabelData(BaseModel):
    sent_im_name_list: str
    sent_bbnum_list: str
    sent_class_list: str
    sent_bbox_list: str

def write_xml(folder, filename, bbox_list):
    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = './images' + filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # Details from first entry
    e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = bbox_list[0]

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = e_width
    SubElement(size, 'height').text = e_height
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list:
        e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = entry

        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = e_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = e_xmin
        SubElement(bbox, 'ymin').text = e_ymin
        SubElement(bbox, 'xmax').text = e_xmax
        SubElement(bbox, 'ymax').text = e_ymax

    # indent(root)
    tree = ElementTree(root)
    output_dir="/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/Annotations"
    xml_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)

def base64str_to_OpenCVImage(rgb_base64):
    rgb = rgb_base64  # raw data with base64 encoding
    decoded_data = base64.b64decode(rgb)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img_rgb = cv.imdecode(np_data,cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    return img_rgb

def predict(im_data,im_info,img_name,back_div,ins_div):
    t3 = time()
    global model
    global score_set
    model.eval()

    gt_boxes = torch.FloatTensor([[1,1,1,1,1]]).cuda()
    num_boxes =torch.FloatTensor([0]).cuda()
    data=im_data,im_info,gt_boxes,num_boxes
    #print(im_data)

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4 * len(classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= data[1][0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    all_boxes = [[[] for _ in xrange(1)]
               for _ in xrange(num_classes)]
    for j in xrange(1, num_classes):
        inds = torch.nonzero(scores[:,j]>0.05).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], 0.01)
            cls_dets = cls_dets[keep.view(-1).long()]
            
            all_boxes[j][0] = cls_dets.cpu().numpy()
        else:
            all_boxes[j][0] = empty_array

        
    bb_num=0
    boxToReturn=[]
    classToReturn=[]
    mean_score=[]
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][0][:, -1]
                                    for j in xrange(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
        else:
            image_thresh = 0.
        for j in xrange(1, num_classes):
            keep = np.where(all_boxes[j][0][:, -1] >= image_thresh)[0]
            if len(keep)>0:
                all_boxes[j][0] = all_boxes[j][0][keep, :]
                real_score=[all_boxes[j][0][d,-1] for d in range(np.minimum(10, all_boxes[j][0].shape[0])) if all_boxes[j][0][d,-1]>0.]
                mean_score.extend(real_score)
                for b_idx in range(len(all_boxes[j][0])):
                    if all_boxes[j][0][b_idx][-1]>0.1:
                        boxToReturn.append(all_boxes[j][0][b_idx][:4].tolist()) 
                        classToReturn.append(class_idx_name[j])
                        bb_num=bb_num+1
                print(all_boxes[j][0])
    if len(mean_score)!=0:
        score_f=np.mean(mean_score)
        score_set.append(div_weight*(back_div_weight*back_div+ins_div_weight*ins_div)+uncertain_weight*(1-score_f))
    else:
        score_set.append(div_weight*(back_div_weight*back_div+ins_div_weight*ins_div)+uncertain_weight)
    predictions = [bb_num,img_name,boxToReturn,classToReturn]#
    t4 = time()
    elapsed2 = t4 - t3
    print('Amount of time taken to make predictions: ' + str(elapsed2) + '\n')
    print(predictions)
    name_prediction_dict[img_name]=predictions

def savemodel(loop_count):
    save_name = os.path.join("/data/ztc/adaptation/Experiment/model/vgg16/cleanlemur","guidance_toysystem_"+method+"_loop"+str(loop_count)+".pth")
    save_checkpoint({
            'session': 1,
            'epoch': 2,
            'model': model.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,}, save_name)
    print('save model: {}'.format(save_name))


def update(loop_count,picked_class_list):
    t3 = time()
    global model
    virtual_finetune_model=finetune(model,cfg,method,training_t_set_name,loop_count,True,1,1,False)
    t1=time()
    final_training_t_set_name = MIR(training_t_set_name,loop_count, old_data_name,retrive_num,virtual_finetune_model,model,method,picked_class_list)
    t2=time()
    print('Amount of time taken to do MIR: ' + str(t2-t1) + '\n')
    model=finetune(model,cfg,method,final_training_t_set_name,loop_count,False,1,1,True)
    t4 = time()
    elapsed2 = t4 - t3
    print('Amount of time taken to update model: ' + str(elapsed2) + '\n')
    #savemodel_thread = threading.Thread(target=savemodel, name="savemodel", args=(loop_count,))
    #savemodel_thread.start()
    savemodel(loop_count)

def LUAL(incom_name_list,score_set):
    #print(incom_name_list)
    #print(score_set)
    sorted_id=sorted(range(len(score_set)), key=lambda k:score_set[k], reverse=False)
    picked_names=[incom_name_list[i] for i in sorted_id][(round(len(incom_name_list)-len(incom_name_list)*label_per)):]
    return picked_names

def systemevaluate(loop_count,initial_model,current_model,training_t_set_name):
    evaluate_model(initial_model,loop_count,training_t_set_name)
    evaluate_model(current_model,loop_count,training_t_set_name)
  
def getins_pred_bbox(im_data, im_info, gt_boxes, num_boxes):

    global model
    model.eval()

    judge=True
    score_f = 0.
    thresh = 0.05
    data=im_data,im_info,gt_boxes,num_boxes

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes)
      
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    all_boxes = [[[] for _ in xrange(1)]
               for _ in xrange(num_classes)]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
          box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    
    pred_boxes /= data[1][0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    vis=True
    for j in xrange(1, num_classes):
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
          # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
            
          all_boxes[j][0] = cls_dets.cpu().numpy()
        else:
          all_boxes[j][0] = empty_array
    
    show_boxes = {j:[] for j in xrange(1, num_classes)}
    max_score=0
    ins_pred_bbox=None
    for j in xrange(1, num_classes):
        #print('class j',j)
        if vis:
            vis_score=[all_boxes[j][0][d,-1] for d in range(np.minimum(10, all_boxes[j][0].shape[0])) if all_boxes[j][0][d,-1]>0.5]
            show_boxes[j]=[all_boxes[j][0][d,:4] for d in range(np.minimum(10, all_boxes[j][0].shape[0])) if all_boxes[j][0][d,-1]>0.5]
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
  

@app.put("/saveLabel")
def saveLabel(s:LabelData):
    now = datetime.datetime.now()
    print("receive flag!"+str(now))
    sent_im_name_list=s.sent_im_name_list.split("!")
    sent_bbnum_list=s.sent_bbnum_list.split("!")
    sent_class_list=s.sent_class_list.split("!")
    sent_bbox_list=s.sent_bbox_list.replace("[", "")
    sent_bbox_list=sent_bbox_list.replace("]", "").split("!")
    print(sent_im_name_list)
    print(sent_bbnum_list)
    print(sent_class_list)
    print(sent_bbox_list)

    global width
    global height

    save_root="Annotations"
    picked_class_list=[]
    entries_by_filename = defaultdict(list)
    for im_idx in range(len(sent_im_name_list)):
        filename=sent_im_name_list[im_idx]+".jpg"
        processed_sent_class_list=sent_class_list[im_idx].lower().split(";")
        print(processed_sent_class_list)
        processed_sent_bbox_list=sent_bbox_list[im_idx].split(";")
        print(processed_sent_bbox_list)
        picked_class_list.append(idx_class_name[processed_sent_class_list[1].upper()])
        for bb_idx in range(int(sent_bbnum_list[im_idx])):
            class_name=processed_sent_class_list[bb_idx+1]
            processed_sent_bbox=processed_sent_bbox_list[bb_idx+1].split(",")
            xmin=processed_sent_bbox[0]
            ymin=processed_sent_bbox[1]
            xmax=processed_sent_bbox[2]
            ymax=processed_sent_bbox[3]
            row=filename, str(width), str(height), class_name, xmin, ymin, xmax, ymax
            entries_by_filename[filename].append(row)  

    for filename, entries in entries_by_filename.items():
        print(filename, len(entries))
        write_xml(save_root, filename, entries)

    print('send labels!')
    global loop_count

    training_t_set=open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+training_t_set_name+str(loop_count)+".txt","w")
    for line in sent_im_name_list:
      training_t_set.write(line+"\n")
    training_t_set.close()

    old_data_set=open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+old_data_name+str(loop_count)+".txt","w")
    with open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+old_data_name+str(loop_count-1)+".txt","r") as f:
      for line in f.readlines():
        old_data_set.write(line)      
    for line in sent_im_name_list:
      old_data_set.write(line+"\n")
    old_data_set.close()

    old_targetdata_set=open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+old_targetdata_name+str(loop_count)+".txt","w")
    with open("/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/ImageSets/Main/"+old_targetdata_name+str(loop_count-1)+".txt","r") as f:
      for line in f.readlines():
        old_targetdata_set.write(line)
    for line in sent_im_name_list:
      old_targetdata_set.write(line+"\n")
    old_targetdata_set.close()

    #evaluate_thread = threading.Thread(target=systemevaluate, name="evaluate", args=(loop_count,initial_model,copy.deepcopy(model),training_t_set_name))
    #evaluate_thread.start()

    loop_count=loop_count+1
    updatecluster_thread = threading.Thread(target=updatecluster, name="updatecluster", args=(loop_count,))
    updatecluster_thread.start()
    #update_thread = threading.Thread(target=update, name="updatemodel", args=(loop_count,picked_class_list))
    #update_thread.start()
    update(loop_count-1,picked_class_list)
    
    
    
    return 1

@app.put("/guidance")
def guidance(d:ImgPathData):
    now = datetime.datetime.now()
    print("receive image!"+str(now))
    img_rgb = base64str_to_OpenCVImage(d.rgb_base64)
    img_name = d.currentTime
    cur_im = im.fromarray(img_rgb, 'RGB')
    cur_im_savename="/root/code/faster-rcnn.pytorch/data/cleanlemur/VOC2007/JPEGImages/"+img_name+".jpg"
    cur_im.save(cur_im_savename)

    global round_cnt
    #global round_num
    global incom_name_list
    global name_prediction_dict
    global pre_im
    global score_set

    t3 = time()
    img_bgr=img_rgb[:,:,::-1].astype(np.float32)
    img_bgr -= np.array([[[102.9801, 115.9465, 122.7717]]])
    im_info= torch.FloatTensor(np.array([[height, width, 1]])).view(1,3).cuda()
    im_data=torch.FloatTensor(img_bgr).permute(2, 0, 1).contiguous().view(1,3, height, width).cuda()
    t1=time()
    print("Time to process input:"+str(t1-t3))
    gt_boxes = torch.FloatTensor([[1,1,1,1,1]]).cuda()
    num_boxes =torch.FloatTensor([0]).cuda()
    ins_score,ins_pred_bbox=getins_pred_bbox(im_data, im_info, gt_boxes, num_boxes)

    back_div,ins_div=judgecluster(im_data,center_list,method,loop_count,pca,normlized_centers,x_min,x_max, ins_center_list,ins_pca,ins_normlized_centers,ins_x_min,ins_x_max,ins_score,ins_pred_bbox)
    
    if (back_div>guidance_threshold) and (ins_div>guidance_threshold):
        
        # predict in background for every possible choosen image
        predict_thread = threading.Thread(target=predict, name="Predictor", args=(im_data,im_info,img_name,back_div,ins_div))
        predict_thread.start()
        incom_name_list.append(img_name)
        round_cnt=round_cnt+1
        if round_cnt==round_num:       
            predict_thread.join()
            print(incom_name_list)
            picked_image_list=LUAL(incom_name_list,score_set)
            print(picked_image_list)
            picked_name_prediction_dict={}
            for picked_name in picked_image_list:
                picked_name_prediction_dict[picked_name]=name_prediction_dict[picked_name]
            user_guidance=picked_name_prediction_dict
            round_cnt=0
            incom_name_list=[]
            name_prediction_dict={}
            score_set=[]
        else:
            user_guidance="0" #means Good job! Please take next photo!
    else:
        if back_div<=guidance_threshold:
            user_guidance="1" # means Sorry! Please adjust your position and retake a photo!
        else:
            user_guidance="2" # means Sorry! Please adjust your pose and retake a photo!

    jsonData = json.dumps(user_guidance)
    print(jsonData)
    print("sending back guidance!"+str(datetime.datetime.now()))
    t4 = time()
    elapsed2 = t4 - t1
    print("Time to judge the guidance:"+str(elapsed2))
    return jsonData

@app.put("/realtimeguidance")
def realtimeguidance(d:ImgPathData):
    now = datetime.datetime.now()
    print("receive image!"+str(now))
    img_rgb = base64str_to_OpenCVImage(d.rgb_base64)
    img_name = d.currentTime
    cur_im = im.fromarray(img_rgb, 'RGB')

    t3 = time()
    img_bgr=img_rgb[:,:,::-1].astype(np.float32)
    img_bgr -= np.array([[[102.9801, 115.9465, 122.7717]]])
    im_info= torch.FloatTensor(np.array([[height, width, 1]])).view(1,3).cuda()
    im_data=torch.FloatTensor(img_bgr).permute(2, 0, 1).contiguous().view(1,3, height, width).cuda()
    t1=time()
    print("Time to process input:"+str(t1-t3))
    gt_boxes = torch.FloatTensor([[1,1,1,1,1]]).cuda()
    num_boxes =torch.FloatTensor([0]).cuda()
    ins_score,ins_pred_bbox=getins_pred_bbox(im_data, im_info, gt_boxes, num_boxes)

    back_div,ins_div=judgecluster(im_data,center_list,method,loop_count,pca,normlized_centers,x_min,x_max, ins_center_list,ins_pca,ins_normlized_centers,ins_x_min,ins_x_max,ins_score,ins_pred_bbox)
    
    user_guidance=str([min(round(back_div,2),1),min(round(ins_div,2),1)])
    jsonData = json.dumps(user_guidance)
    print(jsonData)
    print("sending back real time guidance!"+str(datetime.datetime.now()))
    t4 = time()
    elapsed2 = t4 - t1
    print("Time to judge the guidance:"+str(elapsed2))
    return jsonData

# Run the API with uvicorn
if __name__ == '__main__':
    # INPUT_DIR = "/Users/ashleykwon/Desktop/test/1139155_stock-photo-ring-tailed-lemur_jpg.rf.726b0498d8ce08b2d541ec56748d7a43.jpg"
    uvicorn.run(app, host='172.17.0.6', port=5958)
    #10.197.178.131:51040
    # 172.28.134.97
    # 192.168.1.8 -> I3T 5G
    # 10.197.65.79
    #android may not run on https
