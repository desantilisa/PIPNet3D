#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:48:02 2024

@author: 
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from copy import deepcopy

sys.path.append("/3DPIPNet/src/")
from utils import plot_rgb_slices, get_args, save_args, Log
sys.path.append("/3DPIPNet/src/data/")
from make_dataset import get_dataloaders
sys.path.append("/3DPIPNet/src/models/")
from model_builder import load_trained_pipnet
from test_model import eval_pipnet
from test_model import get_thresholds, eval_ood

sys.path.append("/3DPIPNet/src/visualization/")
from vis_pipnet import visualize_topk



#%% Global Variables

current_fold = 1
uncoherent_ps = {1:[], 2:[276], 3:[193], 4:[225], 5:[378]} # Prototypes index with a Pattern Coherence < 3.5
net = "3Dresnet18"
task_performed = "Test PIPNet"
ps_to_supress = uncoherent_ps[current_fold]

args = get_args(current_fold, net, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Get Dataloaders for the current_fold

dataloaders = get_dataloaders(args)

trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]
    
    
#%% Evaluate 3D-PIPNet trained for the current_fold

print("------", flush = True)
print("PIPNet performances @fold: ", current_fold, flush = True)
    
pipnet = load_trained_pipnet(args)
pipnet.eval()

# Get the latent space dimensions (needed for prototypes' visualization)
with torch.no_grad():
    xs1, _, _ = next(iter(trainloader))
    xs1 = xs1.to(device)
    proto_features, _, _ = pipnet(xs1)
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape # needed for calculating image patch size
    args.hshape = hshape # needed for calculating image patch size
    args.dshape = dshape # needed for calculating image patch size
    print("Output shape: ", proto_features.shape, flush=True)


#%% Get the Global Explanation
topks, img_prototype, proto_coord = visualize_topk(
    pipnet, 
    projectloader, 
    args.num_classes, 
    device, 
    'visualised_prototypes_topk', 
    args,
    save=False,
    )

# set weights of prototypes that are never really found in projection set to 0
set_to_zero = []

if topks:
    for prot in topks.keys():
        found = False
        for (i_id, score) in topks[prot]:
            if score > 0.1:
                found = True
        if not found:
            torch.nn.init.zeros_(pipnet.module._classification.weight[:,prot])
            set_to_zero.append(prot)


#%% Evaluate PIPNet: 
#    - Classification performances, 
#    - Explanations' size

print("------", flush = True)
print("Original model: ", flush=True)
info = eval_pipnet(
    pipnet, 
    testloader, 
    "notused", 
    device)

# OOD Detection
for percent in [95.]:
    print("\nOOD Evaluation for epoch", "not used","with percent of", percent, 
          flush=True)
    _, _, _, class_thresholds = get_thresholds(
        pipnet, testloader, args.epochs, device, percent)
    
    print("Thresholds:", class_thresholds, flush=True)
    
    # Evaluate with in-distribution data
    id_fraction = eval_ood(
        pipnet, testloader, args.epochs, device, class_thresholds)
    print("ID class threshold ID fraction (TPR) with percent", percent, ":", 
          id_fraction, flush=True)
    
    # Evaluate with out-of-distribution data
    ood_args = deepcopy(args)
    _, _, _, _, _, ood_testloader, _, _ = get_dataloaders(ood_args)
    
    id_fraction = eval_ood(
        pipnet, ood_testloader, args.epochs, device, class_thresholds)
    print("class threshold ID fraction (FPR) with percent", percent,":", 
          id_fraction, flush=True) 

    
#%% Controllability
# Suppress incoherent prototypes (set weigth connection to 0)

print("------", flush = True)
print("Suppressed model: ", flush=True)

for prot in ps_to_supress:
    torch.nn.init.zeros_(pipnet.module._classification.weight[:,prot])
    set_to_zero.append(prot)

info = eval_pipnet(
    pipnet, 
    testloader, 
    "notused", 
    device)

# OOD Detection
for percent in [95.]:
    print("\nOOD Evaluation for epoch", "not used","with percent of", percent, 
          flush=True)
    _, _, _, class_thresholds = get_thresholds(
        pipnet, testloader, args.epochs, device, percent)
    
    print("Thresholds:", class_thresholds, flush=True)
    
    # Evaluate with in-distribution data
    id_fraction = eval_ood(
        pipnet, testloader, args.epochs, device, class_thresholds)
    print("ID class threshold ID fraction (TPR) with percent", percent, ":", 
          id_fraction, flush=True)
    
    # Evaluate with out-of-distribution data
    ood_args = deepcopy(args)
    _, _, _, _, _, ood_testloader, _, _ = get_dataloaders(ood_args)
    
    id_fraction = eval_ood(
        pipnet, ood_testloader, args.epochs, device, class_thresholds)
    print("class threshold ID fraction (FPR) with percent", percent,":", 
          id_fraction, flush=True) 