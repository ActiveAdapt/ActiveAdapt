import torch
import numpy as np
from utils import collate_single
import pathlib
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os
from pydantic import BaseModel
from time import time
import ast
import pathlib
import neptune
from torch.utils.data import DataLoader
#import torchvision.ops
from model.roi_layers import nms

from transformations import ComposeSingle, FunctionWrapperSingle, normalize_01, ComposeDouble, FunctionWrapperDouble
from utils import get_filenames_of_path, collate_single, from_dict_to_boundingbox
from faster_RCNN import get_fasterRCNN_resnet
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import heapq

class LemurSpecies(BaseModel):
    image_path: str
    img_base64: str

class LemurModel:
    def __init__(self):
        # parameters
        params = {'EXPERIMENT': 'LEM-149',
                "MODEL_DIR": "/root/code/faster-rcnn.pytorch/finetuneres18_fasterrcnn_wo_finetuneclassifier.pt", # load model from checkpoint
                'OWNER': 'akwon',
                'PROJECT': 'Lemurs-Faster-RCNN-Demo',
                }

        # transformations
        transforms = ComposeSingle([
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
            FunctionWrapperSingle(normalize_01)
        ])

        # import experiment from neptune
        api_key = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOGEzMGQ2Yi0xMzViLTRiNDctYTNkOC0wZDI4NzViMGE3MDUifQ=='  # get the personal api key
        project_name = f'{params["OWNER"]}/{params["PROJECT"]}'
        #project = neptune.init(project_qualified_name=project_name, api_token=api_key)  # get project
        experiment_id = params['EXPERIMENT']  # experiment id
        #experiment = project.get_experiments(id=experiment_id)[0]
        parameters = {'ANCHOR_SIZE': ((32,), (64,), (128,), (256,),), 
                        'ASPECT_RATIOS': ((0.5, 1.0, 2.0),), 
                        'BACKBONE': 'resnet18', 
                        'BATCH_SIZE': 8.0, 
                        'CLASSES': 6.0, 
                        'EXPERIMENT': 'lemurs', 
                        'FPN': True, 
                        'IMG_MEAN': '[0.485, 0.456, 0.406]', 
                        'IMG_STD': '[0.229, 0.224, 0.225]', 
                        'IOU_THRESHOLD': 0.6, 
                        'LR': 0.001, 
                        'MAXEPOCHS': 700.0, 
                        'MAX_SIZE': 1024.0, 
                        'MIN_SIZE': 1024.0, 
                        'PRECISION':32.0, 
                        'PROJECT':'Lemurs-Faster-RCNN-Demo', 
                        'SEED':42.0}

        # properties = experiment.get_properties()
       

        transform = GeneralizedRCNNTransform(min_size=int(parameters['MIN_SIZE']),
                                            max_size=int(parameters['MAX_SIZE']),
                                            image_mean=ast.literal_eval(parameters['IMG_MEAN']),
                                            image_std=ast.literal_eval(parameters['IMG_STD']))

        checkpoint = torch.load(params['MODEL_DIR'], map_location=torch.device('cpu'))

        # model init
        self.model = get_fasterRCNN_resnet(num_classes=int(parameters['CLASSES']),
                                    backbone_name=parameters['BACKBONE'],
                                    anchor_size=parameters['ANCHOR_SIZE'],
                                    aspect_ratios=parameters['ASPECT_RATIOS'],
                                    fpn=True,
                                    min_size=int(parameters['MIN_SIZE']),
                                    max_size=int(parameters['MAX_SIZE'])
                                    )

        # load weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()

    def roundBoxCoordinates(box, maxWidth, maxHeight):
        newBox = [0,0,0,0]
        for i in range(4):
            if i == 0 or i == 3:
                newBox[i] = min(round(box[i]), maxWidth)
            else: 
                newBox[i] = min(round(box[i]), maxHeight)
        return newBox


    def predict_species(self, img, currentTime, t3):
        # transform = GeneralizedRCNNTransform(min_size=int(parameters['MIN_SIZE']),
        #                                     max_size=int(parameters['MAX_SIZE']),
        #                                     image_mean=ast.literal_eval(parameters['IMG_MEAN']),
        #                                     image_std=ast.literal_eval(parameters['IMG_STD']))
        transforms = ComposeSingle([
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
            FunctionWrapperSingle(normalize_01)
        ])

        img = transforms(img)
        img = torch.from_numpy(img).type(torch.float32)
        H = img.size()[1]
        W = img.size()[2]
        print(H, W)
        #topk = 2

        
        class_names = ['BLACK-AND-WHITE-RUFFED-LEMUR','BLUE-EYED-BLACK-LEMUR','COQUERELS-SIFAKA', 'RED-RUFFED-LEMUR', 'RING-TAILED-LEMUR']
        boxes = []
        scores = []

        if (len(img) != 0):
            with torch.no_grad():
                pred = self.model([img])
                t_p = time()
                print("time used to predict", t_p-t3)

                labels = pred[0]['labels']
                boxes = pred[0]['boxes']
                scores = pred[0]['scores']
                print(len(boxes))
                
                indices = nms(boxes, scores, 0.3)
                print(len(indices))
                #print(len(indices), len(torchvision.ops.nms(boxes, scores, 0)))
                scores = [scores[i] for i in indices]
                boxes = [boxes[i] for i in indices]
                labels = [labels[i] for i in indices]
                print(scores)
                predictedclass = [class_names[val-1] for val in labels]
                print(predictedclass)
                
                top_2_idx = np.argsort(scores)#[-topk:]
                print(top_2_idx)
                prob_t = 0.25
                cut = len([i for i in scores if i<=prob_t])
                print(cut)
                
                #print(top_2_idx)
                top_2_idx = top_2_idx[cut:]
                print(top_2_idx)
                topk = len(top_2_idx)

                #maxIdx = scores.index(max(scores))
                newIdx = [labels[pick_i]-1 for pick_i in top_2_idx]
                classToReturn = [class_names[val] for val in newIdx]
                print(classToReturn)
                print([scores[pick_i] for pick_i in top_2_idx])

                boxToReturn = [boxes[pick_i].tolist() for pick_i in top_2_idx]
                print(boxToReturn)
                boxToReturn = [[round(box[0]), round(box[1]), min(round(box[2]), W), min(round(box[3]), H)] for box in boxToReturn]
                print(boxToReturn)


        return [topk, currentTime, boxToReturn, classToReturn]

            
