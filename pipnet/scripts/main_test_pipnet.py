"""
Created on Tue Jan 16 16:18:10 2024

@author: lisadesanti
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
from copy import deepcopy

from utils import get_args
from make_dataset import get_dataloaders
from model_builder import load_trained_pipnet
from test_model import eval_pipnet

from test_model import get_local_explanations
from test_model import get_thresholds, eval_ood
from test_model import eval_local_explanations
from test_model import check_empty_prototypes
from vis_pipnet import visualize_topk



#%% Global Variables

backbone_dic = {1:"resnet3D_18_kin400", 2:"convnext3D_tiny"}

current_fold = 1
net = backbone_dic[1]
task_performed = "test_pipnet"

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
    xs1, _ = next(iter(testloader))
    xs1 = xs1.to(device)
    print("Input shape: ", xs1.shape, flush=True)
    proto_features, _, _ = pipnet(xs1)
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape # needed for calculating image patch size
    args.hshape = hshape # needed for calculating image patch size
    args.dshape = dshape # needed for calculating image patch size
    print("Output shape: ", proto_features.shape, flush=True)


#%% Get the Global Explanation
top1, img_prototype_top1, proto_coord_top1 = visualize_topk(
    pipnet, 
    projectloader, 
    args.num_classes, 
    device, 
    'clinical_feedback_global_explanations', 
    args,
    plot=True,
    save=False,
    k=1)

topks, img_prototype, proto_coord = visualize_topk(
    pipnet, 
    projectloader, 
    args.num_classes, 
    device, 
    'visualised_prototypes_topk', 
    args,
    plot=False,
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
    print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)

print("Classifier weights: ", pipnet.module._classification.weight, flush = True)
print("Classifier weights nonzero: ", pipnet.module._classification.weight[pipnet.module._classification.weight.nonzero(as_tuple=True)], (pipnet.module._classification.weight[pipnet.module._classification.weight.nonzero(as_tuple=True)]).shape, flush = True)
print("Classifier bias: ", pipnet.module._classification.bias, flush = True)

for p in topks.keys(): 
    print(pipnet.module._classification.weight[:,p])

# Print weights and relevant prototypes per class
for c in range(pipnet.module._classification.weight.shape[0]):
    relevant_ps = []
    proto_weights = pipnet.module._classification.weight[c,:]
    
    for p in range(pipnet.module._classification.weight.shape[1]):
        if proto_weights[p]> 1e-3:
            relevant_ps.append((p, proto_weights[p].item()))

    print("Class", c, "(", 
          list(testloader.dataset.class_to_idx.keys())[list(testloader.dataset.class_to_idx.values()).index(c)],
          "):", "has", len(relevant_ps), "relevant prototypes: ", relevant_ps,  flush = True)


#%% Evaluate PIPNet: 
#    - Classification performances, 
#    - Explanations' size
info = eval_pipnet(
    pipnet, 
    testloader, 
    "notused", 
    device)
for elem in info.items():
    print(elem)
    
local_explanations_test, y_preds_test, y_trues_test = get_local_explanations(pipnet, testloader, device, args)


#%% Evaluate the prototypes extracted

columns=["detection_rate", "mean_pcc_d", "mean_pcc_h", "mean_pcc_w", "std_pcc_d", "std_pcc_h", "std_pcc_w", "LC"]

ps_test_evaluation = eval_local_explanations(pipnet, local_explanations_test, device, args)

ps_test_detections = ps_test_evaluation[0]
ps_test_mean_coords = pd.DataFrame(ps_test_evaluation[1]).transpose().round(decimals=2)
ps_test_std_coords = pd.DataFrame(ps_test_evaluation[2]).transpose().round(decimals=2)
ps_test_lc = pd.Series(ps_test_evaluation[3])
avg_ps_consistency = np.nanmean(np.array([h for h in ps_test_evaluation[3].values()]))
eval_proto_test = pd.concat([ps_test_detections, ps_test_mean_coords, ps_test_std_coords, ps_test_lc], axis=1)
eval_proto_test.columns = columns  

empty_ps = check_empty_prototypes(args, pipnet, img_prototype_top1, proto_coord_top1)


#%% Evaluate OOD Detection
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

