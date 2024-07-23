#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:19:24 2023

@author: lisadesanti
"""

import sys
from copy import deepcopy
import torch
import torchvision.models as models
import torch.nn as nn

from utils import get_model_layers
from utils import set_device
from convnext_features import convnext_tiny_3d


def load_blackbox(net, channels, img_shape, out_shape):
    
    """ Instantiate the desidered model """
    
    if net == "resnet3D_18_kin400":
        # ResNet3D-18 pretrained on Kinetics400, 3-channel 3D input
        weights = models.video.R3D_18_Weights.DEFAULT # best available weights 
        model = models.video.r3d_18(weights=weights)
        # Recreate the classifier layer and seed it to the target device
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=512, out_features=out_shape, bias=True))
        
    elif net == "convnext3D_tiny":
        # ConvNeXt3D pretrained on Medical Images (STOIC dataset), 1-channel 3D input
        pretrained_path = "" # path of pretrained model, dowloaded from https://github.com/kiedani/submission_2nd_covid19_competition
        pretrained_mode = "multitaskECCV"
        model = convnext_tiny_3d(
                    pretrained = True,
                    added_dim = 2,
                    init_mode = 'two_g',
                    ten_net = 0,
                    in_chan = channels,
                    use_transformer = False,
                    pretrained_path = pretrained_path,
                    pretrained_mode = pretrained_mode,
                    drop_path = 0.0,
                    datasize = 256 # Shape (HxWxD) of the data is 256x256x256
                )
        
    return model


def load_trained_blackbox(net, channels, img_shape, out_shape, model_folder, current_fold):

    model_path = model_folder + "/fold" + str(current_fold) + ".pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_blackbox(net, channels, img_shape, out_shape)
    net.load_state_dict(torch.load(model_path)) 
    net.to(device)
    net.eval()
    
    return net
    