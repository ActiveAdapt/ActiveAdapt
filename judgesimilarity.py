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

from roi_data_layer.roidb import combined_roidb
from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv
from model.roi_layers import nms
from model.utils.net_utils import save_net, load_net, vis_detections

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.spatial import distance

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True

def judgesimilarity(cur_im,pre_im,pca,x_min,x_max,incom_imdb,i,thresh=0.05):
    
    #### pca similarity
    judge=True
    
    cur_im=cv2.resize(cur_im,(600,600))
    cur_im=cur_im.reshape(1,360000)  
    pca_cur_im=pca.transform(cur_im)
    #print(pca_im.shape)   
    normlized_pca_cur_im=(pca_cur_im - x_min) / (x_max - x_min)

    pre_im=cv2.resize(pre_im,(600,600))
    pre_im=pre_im.reshape(1,360000)
    pca_pre_im=pca.transform(pre_im)
    #print(pca_im.shape)   
    normlized_pca_pre_im=(pca_pre_im - x_min) / (x_max - x_min)
    
    dist=distance.euclidean(normlized_pca_pre_im, normlized_pca_cur_im)
    #print(dist)
    if dist<thresh:
      judge=False
    '''
    #### sift similarity
    judge=True
    cur_im=cv2.imread(incom_imdb.image_path_at(i))
    pre_im=cv2.imread(incom_imdb.image_path_at(i-1))
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(cur_im,None)
    kp2, des2 = sift.detectAndCompute(pre_im,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
      if m.distance < 0.75*n.distance:
        good.append([m])
        a=len(good)
        percent=(a*100)/len(kp2)
        #print("{} % similarity".format(percent))
        if percent >= 60.00:
            judge=False
            break
    
    #### surf similarity
    judge=True
    cur_im=cv2.imread(incom_imdb.image_path_at(i),0)
    pre_im=cv2.imread(incom_imdb.image_path_at(i-1),0)
    orb = cv2.ORB_create(nfeatures=500)

    kp1,des1 = orb.detectAndCompute(cur_im,None)
    kp2,des2 = orb.detectAndCompute(pre_im,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
      if m.distance < 0.75*n.distance:
        good.append([m])
        a=len(good)
        percent=(a*100)/len(kp2)
        #print("{} % similarity".format(percent))
        if percent >= 60.00:
            judge=False
            break
    '''
    
    return judge #True=not similar
