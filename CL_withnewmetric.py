

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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import random
from judgeimage import judgeimage
from finetune import finetune
from mixtrain import mixtrain
from evaluate_model import evaluate_model
from MIR import MIR
from judgecluster import judgecluster
from judgesimilarity import judgesimilarity
from getins_pred_bbox import getins_pred_bbox

from roi_data_layer.roidb import combined_roidb
from roi_da_data_layer.roidb import combined_roidb as da_combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
#from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.measure import compare_ssim as ssim
import pandas as pd

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='indoor', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  # parser.add_argument('--load_dir', dest='load_dir',
  #                     help='directory to load models', default="models",
  #                     type=str)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--model_dir', dest='model_dir',
                      help='directory to load models', default="models.pth",
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
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=6, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--model_name', dest='model_name',
                      help='model file name',
                      default='res101.bs1.pth', type=str)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="/data/ztc/adaptation/Experiment/model",
                      type=str)
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')
  args = parser.parse_args()
  return args

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
  args = parse_args()

  #print('Called with args:')
  #print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  #np.random.seed(cfg.RNG_SEED)
  print('loading our dataset...........')
  '''
  args.s_imdb_name = "cleanlemur_2007_train_s"
  args.t_imdb_name = "cleanlemur_2007_train_t_10fs"
  args.s_imdbtest_name="cleanlemur_2007_test_s"
  args.t_imdbtest_name = "cleanlemur_2007_test_t"
  args.all_imdbtest_name = "cleanlemur_2007_test_all"
  args.s_imdbvalid_name="cleanlemur_2007_valid_s"
  args.t_imdbvalid_name = "cleanlemur_2007_valid_t"
  args.all_imdbvalid_name = "cleanlemur_2007_valid_all"
  '''
  args.s_imdb_name = "indoor_2007_train_mixup_s"
  args.t_imdb_name = "indoor_2007_train_mixup_t_10fs"
  args.s_imdbtest_name="indoor_2007_test_mixup_s"
  args.t_imdbtest_name = "indoor_2007_test_mixup_t"
  args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']


  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  #print('Using config:')
  #pprint.pprint(cfg)

  max_per_image = 3
  if args.cuda:
    cfg.CUDA = True
  

  incom_batch=20
  label_per=0.5
  method="low"
  G=0.9
  B=0.5
  retrive_num=105#100 for lemur#70
  minibatch_flag=True
  dataset_name="indoor"#cleanlemur
  
  training_t_set_name="training_set_t_CL_"+method
  training_t_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+training_t_set_name+"0.txt","w")
  with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_mixup_t_10fs.txt","r") as f:#train_t_10fs
    for line in f.readlines():
      training_t_set.write(line)
  training_t_set.close()

  old_data_name="old_data_set_CL_"+method
  old_data_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_data_name+"0.txt","w")
  with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_mixup_t_10fs.txt","r") as f:#train_t_10fs
    for line in f.readlines():
      old_data_set.write(line)
  with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_mixup_s.txt","r") as f:#train_s
    for line in f.readlines():
      old_data_set.write(line)
  old_data_set.close()

  old_targetdata_name="old_targetdata_set_CL_"+method
  old_targetdata_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_targetdata_name+"0.txt","w")
  with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_mixup_t_10fs.txt","r") as f:#train_t_10fs
    for line in f.readlines():
      old_targetdata_set.write(line)
  old_targetdata_set.close()
  
  error_set_notempty_flag=True
  remaining_set=[]
  with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/train_indoor_cl_round1.txt","r") as f:#incom_round1
    for line in f.readlines():
      line=line.strip('\n')
      remaining_set.append(line)
  

  cfg.TRAIN.USE_FLIPPED = False

  incom_imdb, incom_roidb, incom_ratio_list, incom_ratio_index = combined_roidb(dataset_name+"_2007_train_indoor_cl_round1") #cleanlemur_2007_incom_round1
  incom_imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(incom_roidb)))
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(incom_imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(incom_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(incom_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(incom_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  
  loop_count=0

  
 
  while (loop_count<7):
    incom_set = remaining_set
    loop_count=loop_count+1
    error_set=[]
    score_set=[]
    picked_set = []
    remaining_set=[]
    
    if loop_count==1:
      load_name = "/data/ztc/adaptation/Experiment/model/vgg16/"+dataset_name+"/indoor_ourmixup+FT_10fs_1_5.pth" #finetune_cleanlemur_ours_epoch12_.pth
    else:
      load_name = "/data/ztc/adaptation/Experiment/model/vgg16/"+dataset_name+"/finetune_CL_"+dataset_name+"_"+method+"_loop"+str(loop_count-1)+".pth"
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict({k: v for k, v in checkpoint['model'].items() if k in fasterRCNN.state_dict()})
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print('load model successfully!')
  
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
      im_data = im_data.cuda()
      im_info = im_info.cuda()
      num_boxes = num_boxes.cuda()
      gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
      fasterRCNN.cuda()

    num_images = len(incom_imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(incom_imdb.num_classes)]

    dataset = roibatchLoader(incom_roidb, incom_ratio_list, incom_ratio_index, 1, \
                        incom_imdb.num_classes, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

    ################## center of gray scale of old target data and center of instances
    print('loading lemur dataset.........')
    old_t_imdb_name = dataset_name+"_2007_old_targetdata_set_CL_"+method+str(loop_count-1)#old_data_set_CL_high12"
   
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
      
      img_gray=cv2.cvtColor(old_t_im_data[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2GRAY)
      img=cv2.resize(img_gray,(600,600))
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
        resized_ins=cv2.resize(ins,(600,600))
        resized_ins=resized_ins.reshape(1,360000)  
        ins_vector=resized_ins
        if insNoneFlag==True:
          instance_vector_list=ins_vector
          insNoneFlag=False
        else:
          instance_vector_list=np.concatenate((instance_vector_list,ins_vector),axis=0)

    pca=PCA(n_components=70)#3
    pca_model=pca.fit(vector_list)
    pca_scale=pca.transform(vector_list)
    print(pca_scale.shape)
    #pca_df_scale=pd.DataFrame(pca_scale,columns=['pc1','pc2','pc3'])
    
    #kmeans_pca_scale = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df_scale)
    kmeans_pca_scale = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_scale)
    #print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(pca_df_scale, kmeans_pca_scale.labels_, metric='euclidean')))
    center_list = kmeans_pca_scale.cluster_centers_
    print(center_list)
    x_min, x_max = np.min(pca_scale, 0), np.max(pca_scale, 0)
    normlized_centers = (center_list - x_min) / (x_max - x_min)


    ins_pca=PCA(n_components=70)#
    ins_pca_model=ins_pca.fit(instance_vector_list)
    ins_pca_scale=ins_pca.transform(instance_vector_list)
    print(ins_pca_scale.shape)
    #ins_pca_df_scale=pd.DataFrame(ins_pca_scale,columns=['pc1','pc2','pc3'])
    
    #ins_kmeans_pca_scale = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(ins_pca_df_scale)
    ins_kmeans_pca_scale = KMeans(n_clusters=5, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(ins_pca_scale)
    #print('KMeans PCA Scaled Silhouette Score: {}'.format(silhouette_score(ins_pca_df_scale, ins_kmeans_pca_scale.labels_, metric='euclidean')))
    ins_center_list = ins_kmeans_pca_scale.cluster_centers_
    print(ins_center_list)
    ins_x_min, ins_x_max = np.min(ins_pca_scale, 0), np.max(ins_pca_scale, 0)
    ins_normlized_centers = (ins_center_list - ins_x_min) / (ins_x_max - ins_x_min)
    ##################

    data_iter = iter(dataloader)
    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  
    picked_names=[]
    picked_class_list=[]
    class_list=[]
    InCluster_list=[]
    pre_im=None
    for i in range(num_images):
      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      
      # judge whether the image is predicted wrongly or not
      judge,score_f=judgeimage(fasterRCNN, im_data, im_info, gt_boxes, num_boxes, cfg, args, incom_imdb, data, i, empty_array, all_boxes)
      
      # judge whether the image is already near the center of the cluster
      ins_score,ins_pred_bbox=getins_pred_bbox(fasterRCNN, im_data, im_info, gt_boxes, num_boxes, cfg, args, incom_imdb, data, i, empty_array, all_boxes)
      back_div,ins_div=judgecluster(im_data,center_list,method,loop_count,pca,normlized_centers,x_min,x_max, ins_center_list,ins_pca,ins_normlized_centers,ins_x_min,ins_x_max,ins_score,ins_pred_bbox)

      # judge similarity of adjacent images
      cur_im=cv2.cvtColor(im_data[0].detach().cpu().numpy().transpose(1,2,0),cv2.COLOR_RGB2GRAY)
      if i==0:
        Conj_NotSame=True     
      else:   
        Conj_NotSame = judgesimilarity(cur_im,pre_im,pca,x_min,x_max,incom_imdb,i)
      pre_im=cur_im
      simi=0
      if Conj_NotSame==False:
        simi=-1000
      
      if True:#Conj_NotSame:#NotinCluster and Conj_NotSame:#True:#judge==False:True:#
        error_set.append(incom_imdb.image_path_at(i).split('/')[-1][:-4])
        score_set.append(G*((1-B)*ins_div+B*back_div)+(1-G)*(1-score_f))#(G*((1-B)*ins_div+B*back_div)+(1-G)*(1-score_f))#(0.7*(0.1*ins_div+0.9*back_div)+0.3*(1-score_f))#uncertainty:(score_f)
        class_list.append(round(gt_boxes[0][0,-1].detach().cpu().item()))
      else:
        InCluster_list.append(incom_imdb.image_path_at(i).split('/')[-1][:-4])

    up_toc = time.time()
    #uncertainty is descending
    #sorted_id=sorted(range(len(score_set)), key=lambda k:score_set[k], reverse=True)
    #diversity is ascending
    sorted_id=sorted(range(len(score_set)), key=lambda k:score_set[k], reverse=False)
    if method=="and":
      index_value = random.sample(list(enumerate(error_set)), round(len(error_set)*label_per))
      picked_names.extend([p for _,p in index_value])
      picked_class_list.extend([class_list[p] for p,_ in index_value])
    elif method=="igh":
      picked_names.extend([error_set[i] for i in sorted_id][:round(len(error_set)*label_per)])
      picked_class_list.extend([class_list[p] for p in sorted_id][:round(len(error_set)*label_per)])
    else:
      picked_names.extend([error_set[i] for i in sorted_id][(round(len(error_set)-len(error_set)*label_per)):])
      picked_class_list.extend([class_list[p] for p in sorted_id][(round(len(error_set)-len(error_set)*label_per)):])

    print("picked_names len",len(picked_names))
    print("Incluster len",len(InCluster_list))
    print("picked_class")
    for i in range(incom_imdb.num_classes):
      print(picked_class_list.count(i+1))
    

    training_t_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+training_t_set_name+str(loop_count)+".txt","w")
    for line in picked_names:
      training_t_set.write(line+"\n")
    training_t_set.close()

    
    if torch.cuda.is_available() and not args.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    virtual_finetune_model=finetune(method,training_t_set_name,loop_count,load_name,False,1,1,False)
    final_training_t_set_name = MIR(training_t_set_name,loop_count, old_data_name,retrive_num,virtual_finetune_model,load_name,method,picked_class_list)
    ftepoch=1
    finetune_model=finetune(method,final_training_t_set_name,loop_count,load_name,True,1,ftepoch,minibatch_flag)
    down_tic = time.time()
    update_time = down_tic - up_toc
    print('pick and update time:',update_time)
    for test_i in range(8):
      evaluate_model(finetune_model, test_i+1)

    
    cfg.TRAIN.USE_FLIPPED = False
    incom_imdb, incom_roidb, incom_ratio_list, incom_ratio_index = combined_roidb(dataset_name+"_2007_train_indoor_cl_round"+str(loop_count+1)) #"cleanlemur_2007_incom_round"+str(loop_count+1)
    incom_imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(incom_roidb)))

    old_data_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_data_name+str(loop_count)+".txt","w")
    with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_data_name+str(loop_count-1)+".txt","r") as f:
      for line in f.readlines():
        old_data_set.write(line)      
    for line in picked_names:
      old_data_set.write(line+"\n")
    old_data_set.close()

    old_targetdata_set=open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_targetdata_name+str(loop_count)+".txt","w")
    with open("/root/code/faster-rcnn.pytorch/data/"+dataset_name+"/VOC2007/ImageSets/Main/"+old_targetdata_name+str(loop_count-1)+".txt","r") as f:
      for line in f.readlines():
        old_targetdata_set.write(line)      
    for line in picked_names:
      old_targetdata_set.write(line+"\n")
    old_targetdata_set.close()
    
      


      
      
      
