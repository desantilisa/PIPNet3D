#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:43:20 2023

@author: 
"""

#from util.log import Log
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn


sys.path.append("/3DPIPNet/src/")
from utils import plot_3d_slices
from utils import set_seeds
from utils import set_device
from utils import get_optimizer_nn
from utils import init_weights_xavier
from utils import get_args
from utils import save_args
from utils import Log
sys.path.append("/3DPIPNet/src/data/")
from make_dataset import get_dataloaders
sys.path.append("/3DPIPNet/src/models/")
from pipnet import get_network, PIPNet
from train_model import train_pipnet
from test_model import eval_pipnet
sys.path.append("/3DPIPNet/src/visualization/")
from vis_pipnet import visualize, visualize_topk



#%% Global Variables

current_fold = 1
net = "3Dresnet18"
task_performed = "Train PIPNet"

args = get_args(current_fold, net, task_performed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
 
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

log = Log(args.log_dir)


#%% Train the model

if not os.path.isdir(args.log_dir):
    os.mkdir(args.log_dir)


# Set the device
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
    
net = net.to(device=device)
net = nn.DataParallel(net, device_ids = device_ids)  

optimizer = get_optimizer_nn(net, args)
optimizer_net = optimizer[0]
optimizer_classifier = optimizer[1] 
params_to_freeze = optimizer[2] 
params_to_train = optimizer[3] 
params_backbone = optimizer[4]   

    
# Initialize or load model
with torch.no_grad():
    
    if args.state_dict_dir_net != '':
        
        epoch = 0
        checkpoint = torch.load(
            args.state_dict_dir_net, map_location = device)
        net.load_state_dict(checkpoint['model_state_dict'], strict = True) 
        print("Pretrained network loaded", flush = True)
        net.module._multiplier.requires_grad = False
        
        try:
            optimizer_net.load_state_dict(
                checkpoint['optimizer_net_state_dict']) 
        except:
            pass
        
        if torch.mean(net.module._classification.weight).item() > 1.0 and \
            torch.mean(net.module._classification.weight).item() < 3.0 \
            and torch.count_nonzero(torch.relu(
                net.module._classification.weight-1e-5)).float().item() > \
            0.8*(num_prototypes*args.num_classes): 
                
            print("We assume that the classification layer is not yet \
                  trained. We re-initialize it...", 
                  flush = True) # e.g. loading a pretrained backbone only
            
            torch.nn.init.normal_(
                net.module._classification.weight, 
                mean = 1.0,
                std = 0.1) 
            
            torch.nn.init.constant_(net.module._multiplier, val = 2.)
            print("Classification layer initialized with mean", 
                  torch.mean(net.module._classification.weight).item(), 
                  flush = True)
            
            if args.bias:
                torch.nn.init.constant_(
                    net.module._classification.bias, 
                    val = 0.)
        else:
            if 'optimizer_classifier_state_dict' in checkpoint.keys():
                optimizer_classifier.load_state_dict(
                    checkpoint['optimizer_classifier_state_dict'])
        
    else:
        net.module._add_on.apply(init_weights_xavier)
        torch.nn.init.normal_(
            net.module._classification.weight, 
            mean = 1.0,
            std = 0.1) 
        
        if args.bias:
            torch.nn.init.constant_(
                net.module._classification.bias, 
                val = 0.)
            
        torch.nn.init.constant_(net.module._multiplier, val = 2.)
        net.module._multiplier.requires_grad = False

        print("Classification layer initialized with mean", 
              torch.mean(net.module._classification.weight).item(), 
              flush = True)

# Define classification loss function and scheduler
criterion = nn.NLLLoss(reduction='mean').to(device)

scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_net, 
    T_max = len(trainloader_pretraining)*args.epochs_pretrain, 
    eta_min = args.lr_block/100., 
    last_epoch=-1)

# Forward one batch through the backbone to get the latent output size
with torch.no_grad():
    xs1, _, _ = next(iter(trainloader))
    xs1 = xs1.to(device)
    proto_features, _, _ = net(xs1)
    wshape = proto_features.shape[-1]
    hshape = proto_features.shape[-2]
    dshape = proto_features.shape[-3]
    args.wshape = wshape # needed for calculating image patch size
    args.hshape = hshape # needed for calculating image patch size
    args.dshape = dshape # needed for calculating image patch size
    print("Output shape: ", proto_features.shape, flush=True)

if net.module._num_classes == 2:
    
    # Create a csv log for storing the test accuracy, F1-score, mean train 
    # accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview',
                    'epoch',
                    'test_top1_acc',
                    'test_f1',
                    'almost_sim_nonzeros',
                    'local_size_all_classes',
                    'almost_nonzeros_pooled', 
                    'num_nonzero_prototypes', 
                    'mean_train_acc', 
                    'mean_train_loss_during_epoch')
    
    print("Your dataset only has two classes. Is the number of samples \
          per class similar? If the data is imbalanced, we recommend to \
          use the --weighted_loss flag to account for the imbalance.", 
          flush = True)
        
else:
    
    # Create a csv log for storing the test accuracy (top 1 and top 5), 
    # mean train accuracy and mean loss for each epoch
    print("Create LOG!!")
    log.create_log('log_epoch_overview', 
                    'epoch', 
                    'test_top1_acc', 
                    'test_top5_acc', 
                    'almost_sim_nonzeros', 
                    'local_size_all_classes',
                    'almost_nonzeros_pooled', 
                    'num_nonzero_prototypes', 
                    'mean_train_acc', 
                    'mean_train_loss_during_epoch')


lrs_pretrain_net = []


# 3D-PIPNet Training

#%% PHASE (1): Pretraining Prototypes
for epoch in range(1, args.epochs_pretrain+1):
    for param in params_to_train:
        param.requires_grad = True
    for param in net.module._add_on.parameters():
        param.requires_grad = True
    for param in net.module._classification.parameters():
        param.requires_grad = False
    for param in params_to_freeze:
        param.requires_grad = True  # can be set to False when you want to 
                                    # freeze more layers
    for param in params_backbone:
        param.requires_grad = False # can be set to True when you want to 
                                    # train whole backbone (e.g. if dataset 
                                    # is very different from ImageNet)
    
    print("\nPretrain Epoch", 
          epoch, 
          "with batch size", 
          trainloader_pretraining.batch_size, 
          flush = True)
    
    # Pretrain prototypes
    train_info = train_pipnet(
        net, 
        trainloader_pretraining, 
        optimizer_net, 
        optimizer_classifier, 
        scheduler_net, 
        None, 
        criterion, 
        epoch, 
        args.epochs_pretrain, 
        device, 
        pretrain = True, 
        finetune = False)
    
    lrs_pretrain_net += train_info['lrs_net']
    plt.clf()
    plt.plot(lrs_pretrain_net)
    plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net.png'))
    log.log_values('log_epoch_overview', 
                    epoch, 
                    "n.a.", 
                    "n.a.", 
                    "n.a.", 
                    "n.a.", 
                    "n.a.", 
                    "n.a.", 
                    "n.a.", 
                    train_info['loss'])

if args.state_dict_dir_net == '':
    net.eval()
    torch.save(
        {'model_state_dict': net.state_dict(),
          'optimizer_net_state_dict': optimizer_net.state_dict()},
        os.path.join(os.path.join(args.log_dir, 'checkpoints'), 
                      'net_pretrained'))
    net.train()
    
with torch.no_grad():
    if args.epochs_pretrain > 0:
        print("Visualize top-k")
        topks, img_prototype, proto_coord = visualize_topk(
            net, 
            projectloader, 
            len(args.dic_classes),
            device, 
            'visualised_pretrained_prototypes_topk', 
            args,
            save=False)
 
    
#%% PHASE (2): Training PIPNet

# Re-initialize optimizers and schedulers for second training phase
optimizer = get_optimizer_nn(net, args)
optimizer_net = optimizer[0]
optimizer_classifier = optimizer[1] 
params_to_freeze = optimizer[2] 
params_to_train = optimizer[3] 
params_backbone = optimizer[4] 
        
scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_net, 
    T_max = len(trainloader)*args.epochs, 
    eta_min = args.lr_net/100.)

# Scheduler for the classification layer is with restarts, such that the 
# model can re-active zeroed-out prototypes. Hence an intuitive choice. 
if args.epochs <= 30:
    scheduler_classifier = \
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, 
            T_0 = 5, 
            eta_min = 0.001, 
            T_mult = 1, 
            verbose = False)
else:
    scheduler_classifier = \
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, 
            T_0 = 10, 
            eta_min = 0.001, 
            T_mult = 1, 
            verbose = False)
        
for param in net.module.parameters():
    param.requires_grad = False
for param in net.module._classification.parameters():
    param.requires_grad = True

frozen = True
lrs_net = []
lrs_classifier = []
   
for epoch in range(1, args.epochs + 1): 
                 
    epochs_to_finetune = 3  # during finetuning, only train classification 
                            # layer and freeze rest. usually done for a few 
                            # epochs (at least 1, more depends on size of 
                            # dataset)
    if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or \
                                        args.state_dict_dir_net != ''):
        for param in net.module._add_on.parameters():
            param.requires_grad = False
        for param in params_to_train:
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = False
        for param in params_backbone:
            param.requires_grad = False
        finetune = True
    
    else: 
        finetune = False          
        if frozen:
            # unfreeze backbone
            if epoch > (args.freeze_epochs):
                for param in net.module._add_on.parameters():
                    param.requires_grad = True
                for param in params_to_freeze:
                    param.requires_grad = True
                for param in params_to_train:
                    param.requires_grad = True
                for param in params_backbone:
                    param.requires_grad = True   
                frozen = False
            # freeze first layers of backbone, train rest
            else:
                for param in params_to_freeze:
                    param.requires_grad = True # Can be set to False if you 
                                               # want to train fewer layers 
                                               # of backbone
                for param in net.module._add_on.parameters():
                    param.requires_grad = True
                for param in params_to_train:
                    param.requires_grad = True
                for param in params_backbone:
                    param.requires_grad = False
    
    print("\n Epoch", epoch, "frozen:", frozen, flush = True)  
      
    if (epoch == args.epochs or epoch%30 == 0) and args.epochs > 1:
        
        # Set small weights to zero
        with torch.no_grad():
            torch.set_printoptions(profile = "full")
            
            net.module._classification.weight.copy_(torch.clamp(
                net.module._classification.weight.data - 0.001, min=0.)) 
            
            print("Classifier weights: ", 
                  net.module._classification.weight[
                      net.module._classification.weight.nonzero(
                          as_tuple = True)], 
                  (net.module._classification.weight[
                      net.module._classification.weight.nonzero(
                          as_tuple = True)]).shape, 
                  flush = True)
            
            if args.bias:
                print("Classifier bias: ", 
                      net.module._classification.bias, 
                      flush = True)
                
            torch.set_printoptions(profile = "default")
    
    train_info = train_pipnet(
        net, 
        trainloader, 
        optimizer_net, 
        optimizer_classifier, 
        scheduler_net, 
        scheduler_classifier, 
        criterion, 
        epoch, 
        args.epochs, 
        device, 
        pretrain = False, 
        finetune = finetune)
    
    lrs_net += train_info['lrs_net']
    lrs_classifier += train_info['lrs_class']
    
    
    # Evaluate model

    eval_info = eval_pipnet(net, testloader, epoch, device, log)
    log.log_values(
        'log_epoch_overview', 
        epoch, 
        eval_info['top1_accuracy'], 
        eval_info['top5_accuracy'], 
        eval_info['almost_sim_nonzeros'], 
        eval_info['local_size_all_classes'], 
        eval_info['almost_nonzeros'], 
        eval_info['num non-zero prototypes'], 
        train_info['train_accuracy'], 
        train_info['loss'])
    
    with torch.no_grad():
        net.eval()
        torch.save(
            {'model_state_dict': net.state_dict(),
             'optimizer_net_state_dict': optimizer_net.state_dict(),
             'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, 
            os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                         'net_trained'))

        if epoch%30 == 0:
            net.eval()
            torch.save(
                {'model_state_dict': net.state_dict(), 
                  'optimizer_net_state_dict': optimizer_net.state_dict(), 
                  'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, 
                os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                             'net_trained_%s'%str(epoch)))            
    
        # save learning rate in figure
        plt.clf()
        plt.plot(lrs_net)
        plt.savefig(os.path.join(args.log_dir,'lr_net.png'))
        plt.clf()
        plt.plot(lrs_classifier)
        plt.savefig(os.path.join(args.log_dir,'lr_class.png'))

net.eval()
torch.save(
    {'model_state_dict': net.state_dict(),
     'optimizer_net_state_dict': optimizer_net.state_dict(),
     'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, 
    os.path.join(os.path.join(args.log_dir, 'checkpoints'),
                 'net_trained_last'))

topks, img_prototype, proto_coord = visualize_topk(
    net, 
    projectloader, 
    args.num_classes, 
    device, 
    'visualised_prototypes_topk', 
    args,
    save=False)

# set weights of prototypes that are never really found in projection set to 0
set_to_zero = []

if topks:
    for prot in topks.keys():
        found = False
        for (i_id, score) in topks[prot]:
            if score > 0.1:
                found = True
        if not found:
            torch.nn.init.zeros_(net.module._classification.weight[:,prot])
            set_to_zero.append(prot)
    print(
        "Weights of prototypes",
        set_to_zero, 
        "are set to zero because it is never detected with similarity>0.1 \
            in the training set", 
        flush=True)
        
    eval_info = eval_pipnet(
        net, 
        testloader, 
        "notused" + str(args.epochs),
        device, log)
    
    log.log_values(
        'log_epoch_overview', 
        "notused"+str(args.epochs), 
        eval_info['top1_accuracy'], 
        eval_info['top5_accuracy'], 
        eval_info['almost_sim_nonzeros'], 
        eval_info['local_size_all_classes'], 
        eval_info['almost_nonzeros'], 
        eval_info['num non-zero prototypes'], 
        "n.a.", 
        "n.a.")

print("classifier weights: ", 
      net.module._classification.weight, 
      flush = True)

print("Classifier weights nonzero: ", 
      net.module._classification.weight[
          net.module._classification.weight.nonzero(as_tuple=True)], 
      (net.module._classification.weight[
          net.module._classification.weight.nonzero(as_tuple=True)]).shape, 
      flush=True)

print("Classifier bias: ", 
      net.module._classification.bias, 
      flush=True)

# Print weights and relevant prototypes per class
for c in range(net.module._classification.weight.shape[0]):
    relevant_ps = []
    proto_weights = net.module._classification.weight[c,:]
    
    for p in range(net.module._classification.weight.shape[1]):
        if proto_weights[p]> 1e-3:
            relevant_ps.append((p, proto_weights[p].item()))
    if args.val_split == 0.:
        print("Class", 
              c, 
              "(", 
              list(testloader.dataset.class_to_idx.keys())[
                  list(testloader.dataset.class_to_idx.values()).index(c)],
              "):",
              "has", 
              len(relevant_ps),
              "relevant prototypes: ", 
              relevant_ps, 
              flush=True)
        

