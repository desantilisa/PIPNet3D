#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:29:21 2024

@author: lisadesanti
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

from model_builder import load_trained_blackbox
from test_model import eval_blackbox


#%%

task_performed = "test_blackbox_alzheimer_mri" 
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
out_shape = 1
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

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


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


# Load Data utilizzando la classe Parkinson_Dataset
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
dataset_info = test_dataset.dataset_info

val_labels = val_dataset.img_labels
val_labels_tensor = torch.from_numpy(val_labels)
class_counts = torch.bincount(val_labels_tensor)
class_weights = 1.0 / class_counts.float()
weights = class_weights[val_labels_tensor]
val_sampler = WeightedRandomSampler(weights, len(weights), replacement=False) # Manage the unbalanced dataset
dataset_info = test_dataset.dataset_info

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


net = load_trained_blackbox(net, channels, img_shape, out_shape, model_folder, current_fold)
net.eval()

# Inference
predictions, targets, report = eval_blackbox(
    dataloaders['test'],
    net, 
    dic_classes, 
    experiment_folder="",
    save=False)

print(report, flush = True)

cm = confusion_matrix(predictions, targets)
tp = cm[0][0]
fn = cm[0][1]
fp = cm[1][0]
tn = cm[1][1]

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
acc = accuracy_score(predictions, targets)
bal_acc = balanced_accuracy_score(predictions, targets)
f1 = f1_score(predictions, targets)

print("Accuracy: ", acc, "\n",
      "Sensitivity: ", sensitivity, "\n",
      "Specificity: ", specificity, "\n",
      "Balanced Accuracy:", bal_acc, "\n",
      "f1-score", f1, flush = True) 


