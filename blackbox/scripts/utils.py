
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:29:43 2023

@author: lisadesanti
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import torch
import argparse
import pickle
import random
import torch.optim
from datetime import datetime


def get_nested_folders(image_path):
    folders = []
    while True:
        image_path, folder = os.path.split(image_path)
        if folder:
            folders.insert(0, folder)
        else:
            break
    return folders


def set_seeds(seed: int=42):
    """ 
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    
    
def set_device(args:argparse.Namespace):
    
    gpu_list = torch.cuda # args.gpu_ids.split(',')
    device_ids = []  
    
    if args.gpu_ids!='':
        
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
            
    if not args.disable_cuda and torch.cuda.is_available():
        
        if len(device_ids) == 1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
            
        elif len(device_ids) == 0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush = True)
            device_ids.append(torch.cuda.current_device())
            
        else:
            print("This code should work with multiple GPU's but we didn't \
                  test that, so we recommend to use only 1 GPU.",
                  flush = True)
            device_str = ''
            
            for d in device_ids:
                device_str += str(d)
                device_str += ","
                
            device = torch.device('cuda:' + str(device_ids[0]))
    else:
        
        device = torch.device('cpu')
        
    return device, device_ids
    
    
def plot_loss_curves(results):
    
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "val_acc": [...],
             "test_loss": [...],
             "val_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["val_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def get_model_layers(model):
    
    layer_name_list = []
    for layer_name in model.named_children():
        layer_name_list.append(layer_name[0])
        
    # for layer_name in model.named_modules():
    #     layer_name_list.append(layer_name[0])
        
    return layer_name_list
    
    
def get_hidden_activation(
        model: torch.nn.Module,
        layer_name,
        input_data: torch.Tensor):
    """
    Get intermediate activations PyTorch model
    
    Args:
      model: A target PyTorch model.
      layer_name: Name of the layer of interest.
      input_data: Input passed to the PyToch model
    
    """
    
    activations = {}
    
    def get_hidden_activations(name):
        def hook(module, input, output):
            activations[name] = output
        return hook
    
    # Attach hooks to the layers whose activations you want to capture
    desired_layer = getattr(model, layer_name, None)
    hook = desired_layer.register_forward_hook(
        get_hidden_activations(layer_name))
    
    # Perform a forward pass to capture intermediate activations
    model(input_data)
    
    # Detach the hook after capturing activations
    hook.remove()
    
    # Access the intermediate activations from the 'activations' dictionary
    activation = activations[layer_name]
    
    return activation


def get_activations(model: torch.nn.Module,
                    input_data: torch.Tensor,):
    
    layers_name = get_model_layers(model)
    activations = {}
    
    for layer_name in layers_name:
        activations[layer_name] = get_hidden_activation(
            model, layer_name, input_data)
    return activations
    

def check_unfrozen(model):
    all_unfrozen = True
    for name, param in model.named_parameters():
        if not param.requires_grad:
            all_unfrozen = False
            print(f"Parameter {name} is frozen.")
    if all_unfrozen:
        print("All layers are unfrozen.")
    else:
        print("Not all layers are unfrozen.")
        
