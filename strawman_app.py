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
from unity_strawmanMIR import MIR
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
checkpoint = torch.load("/data/ztc/adaptation/Experiment/model/vgg16/cleanlemur/strawman_initial.pth")

model.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in model.state_dict()})
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
model.cuda()
print("Load model successful")

initial_model=copy.deepcopy(model)
'''
checkpoint = torch.load("/data/ztc/adaptation/Experiment/model/vgg16/cleanlemur/strawman_toysystem_high_loop4.pth")
model.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in model.state_dict()})
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']
model.cuda()
'''
empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
max_per_image=1#1
num_classes=6#6

class_idx_name={1:'BLACK-AND-WHITE-RUFFED-LEMUR',2:'BLUE-EYED-BLACK-LEMUR',3:'COQUERELS-SIFAKA',4:'RED-RUFFED-LEMUR',5:'RING-TAILED-LEMUR'}
idx_class_name={'BLACK-AND-WHITE-RUFFED-LEMUR':1,'BLUE-EYED-BLACK-LEMUR':2,'COQUERELS-SIFAKA':3,'RED-RUFFED-LEMUR':4,'RING-TAILED-LEMUR':5}

round_num=20#50
retrive_num=50#125
round_cnt=0
incom_name_list=[]
name_prediction_dict={}
score_set=[]
label_per=0.5
width=720#1440
height=1480#2960
dataset_name='cleanlemur'#'cleanlemur'

method='high'
loop_count=1

training_t_set_name="training_set_t_CL_"+method
training_t_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+training_t_set_name+"0.txt","w")
with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_t_10fs.txt","r") as f:
  for line in f.readlines():
      training_t_set.write(line)
training_t_set.close()

old_targetdata_name="old_targetdata_set_CL_"+method
old_targetdata_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_targetdata_name+"0.txt","w")
with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_t_10fs.txt","r") as f:
  for line in f.readlines():
    old_targetdata_set.write(line)
old_targetdata_set.close()

old_data_name="old_data_set_CL_"+method
old_data_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_data_name+"0.txt","w")
with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_t_10fs.txt","r") as f:
  for line in f.readlines():
    old_data_set.write(line)
with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_s.txt","r") as f:
  for line in f.readlines():
    old_data_set.write(line)
old_data_set.close()

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
    output_dir="/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/Annotations"
    xml_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)

def base64str_to_OpenCVImage(rgb_base64):
    rgb = rgb_base64  # raw data with base64 encoding
    decoded_data = base64.b64decode(rgb)
    np_data = np.frombuffer(decoded_data, np.uint8)
    img_rgb = cv.imdecode(np_data,cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
    return img_rgb

def predict(im_data,im_info,img_name):
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
        score_set.append(score_f)
    else:
        score_set.append(0.)
    predictions = [bb_num,img_name,boxToReturn,classToReturn]#
    t4 = time()
    elapsed2 = t4 - t3
    print('Amount of time taken to make predictions: ' + str(elapsed2) + '\n')
    print(predictions)
    name_prediction_dict[img_name]=predictions

def savemodel(loop_count):
    save_name = os.path.join("/data/ztc/adaptation/Experiment/model/vgg16/"+dataset_name,"strawman_toysystem_"+method+"_loop"+str(loop_count)+".pth")
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
    sorted_id=sorted(range(len(score_set)), key=lambda k:score_set[k], reverse=True)
    picked_names=[incom_name_list[i] for i in sorted_id][(round(len(incom_name_list)-len(incom_name_list)*label_per)):]
    return picked_names
    
def systemevaluate(loop_count,initial_model,current_model,training_t_set_name):
    evaluate_model(initial_model,loop_count,training_t_set_name)
    evaluate_model(current_model,loop_count,training_t_set_name)

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

    training_t_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+training_t_set_name+str(loop_count)+".txt","w")
    for line in sent_im_name_list:
      training_t_set.write(line+"\n")
    training_t_set.close()

    old_data_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_data_name+str(loop_count)+".txt","w")
    with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_data_name+str(loop_count-1)+".txt","r") as f:
      for line in f.readlines():
        old_data_set.write(line)      
    for line in sent_im_name_list:
      old_data_set.write(line+"\n")
    old_data_set.close()

    old_targetdata_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_targetdata_name+str(loop_count)+".txt","w")
    with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_targetdata_name+str(loop_count-1)+".txt","r") as f:
      for line in f.readlines():
        old_targetdata_set.write(line)      
    for line in sent_im_name_list:
      old_targetdata_set.write(line+"\n")
    old_targetdata_set.close()

    #evaluate_thread = threading.Thread(target=systemevaluate, name="evaluate", args=(loop_count,initial_model,copy.deepcopy(model),training_t_set_name))
    #evaluate_thread.start()

    #update_thread = threading.Thread(target=update, name="updatemodel", args=(loop_count,picked_class_list))
    #update_thread.start()
    update(loop_count,picked_class_list)
    
    loop_count=loop_count+1
    
    return 1

@app.put("/guidance")
def guidance(d:ImgPathData):
    now = datetime.datetime.now()
    print("receive image!"+str(now))
    img_rgb = base64str_to_OpenCVImage(d.rgb_base64)
    img_name = d.currentTime
    cur_im = im.fromarray(img_rgb, 'RGB')
    cur_im_savename="/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/JPEGImages/"+img_name+".jpg"
    cur_im.save(cur_im_savename)

    global round_cnt
    #global round_num
    global incom_name_list
    global name_prediction_dict
    global score_set

    t3 = time()
    img_bgr=img_rgb[:,:,::-1].astype(np.float32)
    img_bgr -= np.array([[[102.9801, 115.9465, 122.7717]]])
    im_info= torch.FloatTensor(np.array([[height, width, 1]])).view(1,3).cuda()
    im_data=torch.FloatTensor(img_bgr).permute(2, 0, 1).contiguous().view(1,3, height, width).cuda()
    t1=time()
    print("Time to process input:"+str(t1-t3))
    t4 = time()
    elapsed2 = t4 - t1
    print("Time to judge the guidance:"+str(elapsed2))
    predict_thread = threading.Thread(target=predict, name="Predictor", args=(im_data,im_info,img_name))
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
        user_guidance="0"
    jsonData = json.dumps(user_guidance)
    print(jsonData)
    print("sending back guidance!"+str(datetime.datetime.now()))
    return jsonData
# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='172.17.0.3', port=5958)
