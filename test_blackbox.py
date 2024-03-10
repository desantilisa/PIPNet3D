#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:29:21 2024

@author: 
"""


import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

sys.path.append("/3DPIPNet/src/")
from utils import plot_3d_slices, get_args, save_args, Log
sys.path.append("3DPIPNet/src/data/")
from make_dataset import get_dataloaders
sys.path.append("/3DPIPNet/src/models/")
from model_builder import load_trained_blackbox
from test_model import eval_blackbox


#%%

current_fold = 1
net = "3Dresnet18"
task_performed = "Test Blackbox"

args = get_args(current_fold, net, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


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


net = load_trained_blackbox(args)
net.eval()

# Inference
y_pred, y_true, report = eval_blackbox(
    testloader, 
    net, 
    args.dic_classes, 
    args.experiment_folder,
    save=False)

print(report, flush = True)

cm = confusion_matrix(y_true, y_pred)
tp = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tn = cm[1][1]

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy: ", acc, "\n",
      "Sensitivity: ", sensitivity, "\n",
      "Specificity: ", specificity, "\n",
      "Balanced Accuracy:", bal_acc, "\n",
      "f1-score", f1, flush = True) 


