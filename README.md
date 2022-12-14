# Continual Environmental Adaptation in the Wild under Real-time Data Acquisition Guidance

This repository contains the research artifacts for "Continual Environmental Adaptation in the Wild under Real-time Data Acquisition Guidance", including the implementation of the object detection adaptation system, our own constructed datasets, user study setting for each round, and the system test set collection details.

## Outline

* [Overview](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#1-overview) 
* [Datasets](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#2-datasets) 
* [User study setting for each round](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#3-user-study-setting-for-each-round) 
* [Test set collection details](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#4-test-set-collection-details) 
* [Implementation](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#5-implementation) 

The rest of the repository is organized as follows. [Section 1](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#1-overview) gives a brief overview of ActiveAdapt. [Section 2](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#2-datasets) introduces our own constructed datasets. [Section 3](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#3-user-study-setting-for-each-round) shows the user study setting for each round. [Section 4](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#4-test-set-collection-details) introduces the test set collection details. [Section 5](https://github.com/ActiveAdapt/ActiveAdapt/edit/master/README.md#5-implementation) brief introduces the implementation of ActiveAdapt.

## 1. Overview

<div align="center">
<img src="https://user-images.githubusercontent.com/119776995/207464425-f39a2dd9-c93e-4b87-990d-2bf7591269b1.png" width="500">
</div>
           
We present ActiveAdapt, the first object detection adaptation system that adapts an object detection model to the target domain with small human effort via continually learning from the data streams under real-time data acquisition guidance. We first augment a few target domain samples to enrich features of target environments and object instances for model initialization. Furthermore, we engage users by guiding them in real time to collect diverse data, and supporting data labeling by providing suggested annotations and selecting informative and diverse data to be labeled to save human effort. 

## 2. Datasets

Our collected lemur dataset and constructed indoor dataset are shown in the "/data" directory. (Full dataset will be released later.)

<div align="center">
<img src="https://user-images.githubusercontent.com/119776995/206885991-81291667-94c8-415d-874f-65b6e07d5493.png">
<img src="https://user-images.githubusercontent.com/119776995/206886003-1279678e-3b1f-492c-b502-364609c0f1f1.png">
</div>

##### 2.1 Lemur dataset
We amassed a dataset with 2,534 lemur images sourced from a local wildlife center, YouTube videos, and two image search platforms, Flickr and Wikimedia Commons. The dataset has 5 classes corresponding to lemur species in the wildlife center: blue-eyed black lemurs, ring-tailed lemurs, black-and-white ruffed lemurs, coquerel’s sifakas, and red ruffed lemurs. The dataset includes 2 domains: source domain with 1,267 images of lemurs that are clearly visible in well-lit environments without cages (902, 240, and 125 in the training, validation, and test sets, correspondingly), and target domain including 1,267 images of lemurs (889, 260, and 118 in the training, validation, and test sets, correspondingly) that are blocked by cages (1,140), in motion resulting in blurry images (71), or in dark environments (56).

##### 2.2 Indoor dataset
We include seven classes, which commonly appear in the indoor environment, in the dataset: mobile phones, scissors, light bulbs, cans, balls, mugs and remote controls. [CORe50](https://vlomonaco.github.io/core50/) is chosen to be the source domain dataset in which the images have plain backgrounds. [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html) is chosen to be the target domain dataset in which the images contain complex backgrounds, varying object sizes and occluded objects, which can simulate more realistic settings. We randomly select 4200 samples (2100 target domain samples) to form our indoor dataset, and split it to training set (66%), validation set (17%) and test set (17%)

##### 2.3 Dataset structure
```bash
data
├── cleanlemur
│   ├── VOC2007
│   │   ├── Annotations
│   │   │   ├── source_xxx.xml
│   │   │   ├── ...
│   │   │   ├── target_xxx.xml
│   │   │   └── ...
│   │   ├── ImageSets
│   │   │   ├── Main
│   │   │   │   ├── train_s.txt
│   │   │   │   ├── train_t.txt
│   │   │   │   ├── valid_s.txt
│   │   │   │   ├── valid_t.txt
│   │   │   │   ├── test_s.txt
│   │   │   └── └── test_t.txt
│   │   ├── JPEGImages
│   │   │   ├── source_xxx.jpg
│   │   │   ├── ...
│   │   │   ├── target_xxx.jpg
│   └── └── └── ...
├── indoor
│   ├── VOC2007
│   │   ├── Annotations
│   │   │   ├── source_xxx.xml
│   │   │   ├── ...
│   │   │   ├── target_xxx.xml
│   │   │   └── ...
│   │   ├── ImageSets
│   │   │   ├── Main
│   │   │   │   ├── train_s.txt
│   │   │   │   ├── train_t.txt
│   │   │   │   ├── valid_s.txt
│   │   │   │   ├── valid_t.txt
│   │   │   │   ├── test_s.txt
│   │   │   └── └── test_t.txt
│   │   ├── JPEGImages
│   │   │   ├── source_xxx.jpg
│   │   │   ├── ...
│   │   │   ├── target_xxx.jpg
└── └── └── └── ...
```

## 3. User study setting for each round

<div align="center">
<img src="https://user-images.githubusercontent.com/119776995/206885901-3373c002-241b-41c2-a67a-a556d591859c.png">
</div>

To simulate the realistic class-bias CL data streams, during each round, the user takes images of objects from two classes. 20 images are taken per round, with different numbers of images for each class. Rounds differ, with different scenes and different object poses

## 4. Test set collection details

We pre-collect two diverse datasets containing images with different object poses and backgrounds for system evaluation, since it is inefficient and time-consuming for users in the wild to collect and label the test set in each round required in the CL evaluation. In addition, even if users collect their own test set in each round, the traditional CL metrics obtained by different users cannot be compared fairly. So we pre-collect the test sets to evaluate the model and compare the results fairly.

To have similar image numbers in two test sets, we collect and label 120 images for each class (480 in total) to construct the test dataset for the lemur scenario, and 50 images for each class (450 in total) to form the test dataset for the indoor object scenario.

## 5. Implementation

##### 5.1 Installation           
For the project installation, please refer to https://github.com/jwyang/faster-rcnn.pytorch to get detailed prerequisites.

##### 5.2 For the BiMix initialization
Please train the model using this command:
```
CUDA_VISIBLE_DEVICES=0 python fs_da_trainval_net_ourmixup.py --dataset "dataset" --net vgg16 --s_bs 3 --t_bs 1 --lr 1e-3 --lr_decay_step 6 --cuda --epochs 25
```

##### 5.3 For the detection assistance on the server
Please run the detection assistance on the server:
```
uvicorn app_guidance:app --reload --host "Your IP address" --port "Your port"
```

##### 5.4 For the model testing
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset "dataset" --part "test set name" --model_dir "path to your model" --cuda
```
