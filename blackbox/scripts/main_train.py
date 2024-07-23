#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:52:08 2024

@author: lisadesanti
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

from monai.transforms import (
    Compose,
    Resize,
    RandRotate,
    Affine,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandZoom,
    RandFlip,
    RepeatChannel,
    RandSpatialCrop,
)

from make_dataset import BrainDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from plot_utils import plot_3d_slices

from utils import set_seeds
from utils import plot_loss_curves
from model_builder import load_blackbox
from train_model import train_blackbox
from test_model import eval_blackbox


#%% Global Variables

task_performed = "train_blackbox_alzheimer_mri" 
current_fold = 1
backbone_dic = {1:"resnet3D_18_kin400", 2:"convnext3D_tiny"}
net = backbone_dic[1]

root_folder = "/home/lisadesanti/DeepLearning/ADNI/PIPNet3D/blackbox"
net_dic = {"resnet3D_18_kin400":3, "convnext3D_tiny":1} # 
n_fold = 5           # Number of fold
test_split = 0.2
seed = 42            # seed for reproducible shuffling

dic_classes = {"CN":0, "AD":1}
num_classes = len(dic_classes)
out_shape = num_classes
classification_task = "binary/" 
model_folder = os.path.join(root_folder, "models/", classification_task, net)
downscaling = 2
rows = int(229/downscaling)
cols = int(193/downscaling)
slices = int(160/downscaling)

batch_size = 24
lr = 0.0001
weight_decay = 0.1
gamma = 0.1
step_size = 7
epochs = 100

img_shape = (slices, rows, cols)
channels = net_dic[net]
experiment_folder = os.path.join(root_folder, "runs/", task_performed, classification_task, net, "fold_" + str(current_fold), "Experiment_" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

training_curves_path = os.path.join(experiment_folder, 'training.png')
best_weights_path = os.path.join(experiment_folder, 'best_model.pth')
hyperparameters_file = os.path.join(experiment_folder, 'hyperparameters.json')
report_file = os.path.join(experiment_folder, 'classification_report.txt')

# Hyperparameters
hyperparameters = {"Learning Rate" : lr,
                   "Weight Decay" : weight_decay,
                   "Gamma" : gamma,
                   "Step size" : step_size,
                   "Batch Size" : batch_size,
                   "Epochs" : epochs, "Training Time" : 0}
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Create Image Dataset and Get Dataloaders

# Data augmentation (on-the-fly) parameters
aug_prob = 0.5
rand_rot = 10                       # random rotation range [deg]
rand_rot_rad = rand_rot*math.pi/180 # random rotation range [rad]
rand_noise_std = 0.01               # std random Gaussian noise
rand_shift = 5                      # px random shift
min_zoom = 0.9
max_zoom = 1.1

# Define on-the-fly data augmentation transformation
data_transforms = {
    'train': Compose([
        Resize(spatial_size=img_shape),
        #RandSpatialCrop(roi_size=img_shape),
        RandRotate(range_x=rand_rot_rad, range_y=rand_rot_rad, range_z=rand_rot_rad, prob=aug_prob),
        RandGaussianNoise(std=rand_noise_std, prob=aug_prob),
        #RandGaussianSmooth(prob=aug_prob),
        Affine(translate_params=(rand_shift, rand_shift, rand_shift), image_only=True),
        RandZoom(min_zoom=min_zoom, max_zoom=max_zoom, prob=aug_prob),
        #RandFlip(prob=aug_prob),
        RepeatChannel(repeats=channels),]),
    'val': Compose([Resize(spatial_size = img_shape), RepeatChannel(repeats=channels),]),
    'test': Compose([Resize(spatial_size = img_shape), RepeatChannel(repeats=channels),]),
    }


# Load Data 
train_dataset = BrainDataset(
    dataset_type = task_performed,
    dic_classes = dic_classes,
    set_type='train', 
    transform = data_transforms,
    n_fold = n_fold, 
    current_fold = current_fold)

val_dataset = BrainDataset(
    dataset_type = task_performed,
    dic_classes = dic_classes,
    set_type='val', 
    transform = data_transforms,
    n_fold = n_fold, 
    current_fold = current_fold)

test_dataset = BrainDataset(
    dataset_type = task_performed,
    dic_classes = dic_classes,
    set_type='test', 
    transform = data_transforms,
    n_fold = n_fold, 
    current_fold = current_fold)


train_labels = train_dataset.img_labels
train_labels_tensor = torch.from_numpy(train_labels)
class_counts = torch.bincount(train_labels_tensor)
class_weights = 1.0 / class_counts.float()
weights = class_weights[train_labels_tensor]
train_sampler = WeightedRandomSampler(weights, len(weights), replacement=False) # Manage the unbalanced dataset

val_labels = val_dataset.img_labels
val_labels_tensor = torch.from_numpy(val_labels)
class_counts = torch.bincount(val_labels_tensor)
class_weights = 1.0 / class_counts.float()
weights = class_weights[val_labels_tensor]
val_sampler = WeightedRandomSampler(weights, len(weights), replacement=False) # Manage the unbalanced dataset

test_labels = test_dataset.img_labels
test_labels_tensor = torch.from_numpy(test_labels)
class_counts = torch.bincount(test_labels_tensor)
class_weights = 1.0 / class_counts.float()
weights = class_weights[test_labels_tensor]
test_sampler = WeightedRandomSampler(weights, len(weights), replacement=False) # Manage the unbalanced dataset

dataset_info = train_dataset.dataset_info

# dizionario che contiene le dimensioni dei set
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
dataloaders = {
    'train': DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler),
    'val': DataLoader(dataset=val_dataset, batch_size=1, sampler=val_sampler),
    'test': DataLoader(dataset=test_dataset, batch_size=1, sampler=test_sampler)}

dataiter = iter(dataloaders['test'])
data = next(dataiter)
image, labels = data
plot_3d_slices(np.array(image[0,0,:,:,:]))
    

#%% Get the model

model = load_blackbox(net, channels, img_shape, out_shape)
model.to(device)
# Set the seeds
set_seeds(seed)

# Define the optimization process
# Loss function
criterion = nn.CrossEntropyLoss() 
# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
# Decay LR by a factor of *gamma* every *step_size* epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
        
    
#%% Train the model and save the best weights

if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)

since = time.time()

results = train_blackbox(
    model = model,
    train_dataloader=dataloaders['train'],
    val_dataloader=dataloaders['val'],
    optimizer = optimizer,
    scheduler = exp_lr_scheduler,
    loss_fn = criterion,
    epochs = epochs,
    device = device,
    results_path = experiment_folder)

time_elapsed = time.time() - since
plot_loss_curves(results)
plt.savefig(training_curves_path)
hyperparameters["Training Time"] = time_elapsed
    
    
#%% Test the trained model

# Load best model

loaded_model = load_blackbox(net, channels, img_shape, out_shape)
loaded_model.load_state_dict(torch.load(best_weights_path)) 
loaded_model.to(device)
loaded_model.eval()

# Inference
predictions, targets, report = eval_blackbox(
    dataloaders['test'],
    loaded_model, 
    dic_classes, 
    experiment_folder)

# Save dictionaries

# Save training hyperparameters
with open(hyperparameters_file, 'w') as json_file:
    json.dump(hyperparameters, json_file)

# Save the classification report to a text file
with open(report_file, 'w') as json_file:
    json_file.write(report)
    
# Save the model

if not os.path.exists(model_folder):
    os.makedirs(model_folder)
    
model_path = model_folder + "/fold" + str(current_fold) + ".pth"
torch.save(model.state_dict(), model_path)        






















