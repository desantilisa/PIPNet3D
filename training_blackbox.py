#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:52:08 2024

@author: 
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


sys.path.append("/3DPIPNet/src/")
from utils import set_seeds
from utils import plot_loss_curves
from utils import get_args
sys.path.append("/3DPIPNet/src/data/")
from make_dataset import get_dataloaders
sys.path.append("/3DPIPNet/src/models/")
from model_builder import load_blackbox
from train_model import train_blackbox
from test_model import eval_blackbox


#%% Global Variables

current_fold = 1
net = "3Dresnet18"
task_performed = "Train Blackbox"

args = get_args(current_fold, net, task_performed)
 
training_curves_path = os.path.join(args.experiment_folder, 'training.png')
best_weights_path = os.path.join(args.experiment_folder, 'best_model.pth')
hyperparameters_file = os.path.join(args.experiment_folder, 
                                    'hyperparameters.json')
report_file = os.path.join(args.experiment_folder, 'classification_report.txt')

# Hyperparameters

hyperparameters = {"Learning Rate" : args.lr,
                   "Weight Decay" : args.weight_decay,
                   "Gamma" : args.gamma,
                   "Step size" : args.step_size,
                   "Batch Size" : args.batch_size,
                   "Epochs" : args.epochs,
                   "Training Time" : 0}
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Get Dataloaders

dataloaders = get_dataloaders(args)

trainloader = dataloaders[0]
trainloader_pretraining = dataloaders[1]
trainloader_normal = dataloaders[2] 
trainloader_normal_augment = dataloaders[3]
projectloader = dataloaders[4]
valloader = dataloaders[5]
testloader = dataloaders[6] 
test_projectloader = dataloaders[7]


#%% Get the model

model = load_blackbox(
    net = args.net,
    input_channels = args.channels,
    out_shape = args.out_shape,
    seed = args.seed,)

model.to(device)
# Set the seeds
set_seeds(args.seed)

# Define the optimization process

# Loss function
criterion = nn.BCEWithLogitsLoss() if args.num_classes == 2 else \
            nn.CrossEntropyLoss()
            
# Observe that all parameters are being optimized
optimizer = optim.Adam(
    model.parameters(), 
    lr = args.lr, 
    weight_decay = args.weight_decay)

# Decay LR by a factor of *gamma* every *step_size* epochs
exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer, 
    step_size = args.step_size, 
    gamma = args.gamma)
        
    
#%% Train the model and save the best weights

if not os.path.exists(args.experiment_folder):
    os.makedirs(args.experiment_folder)

since = time.time()

results = train_blackbox(
    model = model,
    train_dataloader = trainloader_normal_augment,
    val_dataloader = valloader,
    optimizer = optimizer,
    scheduler = exp_lr_scheduler,
    loss_fn = criterion,
    epochs = args.epochs,
    device = device,
    results_path = args.experiment_folder)

time_elapsed = time.time() - since

plot_loss_curves(results)

plt.savefig(training_curves_path)

hyperparameters["Training Time"] = time_elapsed
    
    
#%% Test the trained model

# Load best model

loaded_model = load_blackbox(
    net = args.net,
    input_channels = args.channels,
    out_shape = args.out_shape,
    seed = args.seed,)

loaded_model.load_state_dict(torch.load(best_weights_path)) 
loaded_model.to(device)
loaded_model.eval()


# Inference
predictions, targets, report = eval_blackbox(
    testloader, 
    loaded_model, 
    args.dic_classes, 
    args.experiment_folder)

# Save dictionaries

# Save training hyperparameters
with open(hyperparameters_file, 'w') as json_file:
    json.dump(hyperparameters, json_file)

# Save the classification report to a text file
with open(report_file, 'w') as json_file:
    json_file.write(report)
        
