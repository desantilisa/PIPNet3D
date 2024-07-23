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
from utils import get_optimizer_nn
from pipnet import get_network, PIPNet
from convnext_features import convnext_tiny_3d


    
def load_trained_pipnet(args):
    
    models_folder = "/home/lisadesanti/DeepLearning/ADNI/PIPNet3D/pipnet/models/binary/"
    model_path = models_folder + args.net + "/best_pipnet_fold%s"%str(args.current_fold)

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
    net_trained_last = torch.load(model_path)
    net.load_state_dict(net_trained_last['model_state_dict'], strict=True)
    net.to(device)
    net.eval()
    
    return net














