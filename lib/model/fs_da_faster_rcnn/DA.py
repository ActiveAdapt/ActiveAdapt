from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
import torch.nn as nn
from torch.autograd import Function
from model.fs_da_faster_rcnn.LabelResizeLayer import ImageLabelResizeLayer
from model.fs_da_faster_rcnn.LabelResizeLayer import InstanceLabelResizeLayer



class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA(nn.Module):
    def __init__(self,dim):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        #print(self.dim)
        self.Conv1 = nn.Conv2d(self.dim, 1024, kernel_size=3, stride=1,bias=False)
        #self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.reLu=nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1)
        #self.LabelResizeLayer=ImageLabelResizeLayer()

    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        #print(x.size())
        x=self.avgpool(x)
        #print(x.size())
        x=x.view(x.size(0),-1)
        #print(x.size())
        x=F.sigmoid(self.fc(x))
        #print(x.size())
        return x


class _InstanceDA(nn.Module):
    def __init__(self, num_classes):
        super(_InstanceDA,self).__init__()
        self.dc_ip1 = nn.Linear(8192, 1024)
        #self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.4)

        self.dc_ip2 = nn.Linear(1024, 2)
        #self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        #self.clssifer=nn.Linear(1024,1)
        #self.LabelResizeLayer=InstanceLabelResizeLayer()

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_drop1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        return x


