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

sys.path.append("/3DPIPNet/src/")
from utils import get_model_layers
from utils import set_device
from utils import get_optimizer_nn
sys.path.append("/3DPIPNet/src/models/")
from pipnet import get_network, PIPNet


def load_blackbox(
        net:str,
        input_channels:int,
        out_shape:int,
        seed:int):
    
    """ Instantiate the desidered model """
    
    if net == "3Dresnet18":
        weights = models.video.R3D_18_Weights.DEFAULT # best available weights 
        model = models.video.r3d_18(weights=weights)
                
        # Set the manual seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Recreate the classifier layer and seed it to the target device
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=512, out_features=out_shape, bias=True))
        
    return model


def load_trained_blackbox(args):
    
    models_folder = "/3DPIPNet/models/3D-ResNet18/"
    model_path = models_folder + "net_trained_fold" + str(args.current_fold) + ".pth"
    
    device, device_ids = set_device(args)
    
    net = load_blackbox(
        net = args.net,
        input_channels = args.channels,
        out_shape = args.out_shape,
        seed = args.seed)
    
    net.load_state_dict(torch.load(model_path)) 
    net.to(device)
    net.eval()
    
    return net
    
    
def load_trained_pipnet(args):
    

    models_folder = "/3DPIPNet/models/3D-PIPNet_3D-ResNet18/"
    model_path = models_folder + "net_trained_last_fold" + str(args.current_fold)

    device, device_ids = set_device(args)
     
    # Create 3D-PIPNet
    
    network_layers = get_network(args.out_shape, args)
    feature_net = network_layers[0]
    add_on_layers = network_layers[1]
    pool_layer = network_layers[2]
    classification_layer = network_layers[3]
    num_prototypes = network_layers[4]
    
    net = PIPNet(
        num_classes = args.out_shape,
        num_prototypes = num_prototypes,
        feature_net = feature_net,
        args = args,
        add_on_layers = add_on_layers,
        pool_layer = pool_layer,
        classification_layer = classification_layer
        )
    
    net = net.to(device = device)
    net = nn.DataParallel(net, device_ids = device_ids)  
    
    # Load trained network
    net_trained_last = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(net_trained_last['model_state_dict'], strict=True)
    net.to(device)
    net.eval()
    
    return net


 













