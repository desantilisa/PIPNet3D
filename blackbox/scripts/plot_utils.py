#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:16:01 2024

@author: lisadesanti
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import SimpleITK as sitk
import math
import torch
import argparse
import pickle
import random
import torch.optim
from datetime import datetime



def generate_rgb_array(n):
    
    # Generate a list of n distinct colors using matplotlib's 'tab10' colormap
    cmap = plt.get_cmap('tab10')
    color_list = [cmap(i)[:3] for i in np.linspace(0, 1, n)]

    # Convert the list to a NumPy array
    rgb_array = np.array(color_list)

    return rgb_array 

  
def plot_3d_slices(data, 
                   num_columns=10, 
                   cmap="gray", 
                   title=False, 
                   data_min=False, 
                   data_max=False,
                   save_path=False,
                   bottom=False):
    
    depth = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    
    if not(data_min) or not(data_max):
        data_min = data.min()
        data_max = data.max()
    
    r, num_rows = math.modf(depth/num_columns)
    num_rows = int(num_rows)
    if num_rows == 0:
        num_columns = int(r*num_columns)
        num_rows +=1
        r = 0
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows))
        add = np.zeros((new_im, width, height), dtype=type(data[0,0,0]))
        data = np.concatenate((data, add), axis=0)
        num_rows +=1
    
    data = np.reshape(data, (num_rows, num_columns, width, height))

    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    );
        
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img = axarr[i, j].imshow(data[i][j], cmap=cmap, vmin=data_min, vmax=data_max)
                axarr[i, j].axis("off");
            else:
                img = axarr[j].imshow(data[i][j], cmap=cmap, vmin=data_min, vmax=data_max)
                axarr[j].axis("off");

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
    
    if title:
        f.suptitle(title)
            
    # cbar_ax = f.add_axes([0.92, 0, 0.015, 0.9])         
    # f.colorbar(img, cax=cbar_ax)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    

def plot_rgb_slices(
        data, 
        num_columns=10,
        title=False, 
        data_min=False,
        data_max=False,
        save_path=False,
        bottom=False,
        legend=False):
    """
    Plot all the slices of a 3D volume (both gray-scale or RGB) stored in a 
    numpy array.
    Takes:
        - data: np.array, expected dimension (channels, slices, rows, columns)
        - num_columns
        - title
        - data_min
        - data_max
        - save_path
    """
    
    channels, depth, width, height = data.shape
    
    if not(data_min) or not(data_max):
        data_min = data.min()
        data_max = data.max()
    
    r, num_rows = math.modf(depth/num_columns)
    num_rows = int(num_rows)
    if num_rows == 0:
        num_columns = int(r*num_columns)
        num_rows +=1
        r = 0
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows))
        add = np.zeros((channels, new_im, width, height), dtype=type(data[0,0,0,0]))
        data = np.concatenate((data, add), axis=1)
        num_rows +=1
    
    data = np.reshape(data, (channels, num_rows, num_columns, width, height))
    data = np.transpose(data, (1, 2, 3, 4, 0))

    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    );
        
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img = axarr[i, j].imshow(data[i][j], vmin=data_min, vmax=data_max)
                axarr[i, j].axis("off");
            else:
                img = axarr[j].imshow(data[i][j], vmin=data_min, vmax=data_max)
                axarr[j].axis("off");

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
    
    if title:
        if bottom:
            f.suptitle(title, fontsize="large", y=0., va="top", color="gray")
        else:
            f.suptitle(title)
            
    if save_path:
        f.savefig(save_path, bbox_inches='tight')
        
    plt.show()


def plot_atlas_overlay(data, data_atlas="subcortical", num_columns=10, title=False,):
    
    if data_atlas == "subcortical":
        atlas_path = "/home/lisadesanti/DeepLearning/PPMI/Atlases/HarvardOxford-sub-maxprob-thr50-2mm.nii"
    else:
        atlas_path = "/home/lisadesanti/DeepLearning/PPMI/Atlases/HarvardOxford-cort-maxprob-thr50-2mm.nii"
    
    atlas_image = sitk.ReadImage(atlas_path)
    data_atlas = sitk.GetArrayFromImage(atlas_image)
    
    depth = data.shape[0]
    width = data.shape[1]
    height = data.shape[2]
    
    r, num_rows = math.modf(depth/num_columns)
    num_rows = int(num_rows)
    if num_rows == 0:
        num_columns = int(r*num_columns)
        num_rows +=1
        r = 0
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows))
        add = np.zeros((new_im, width, height), dtype=type(data[0,0,0]))
        data = np.concatenate((data, add), axis=0)
        data_atlas = np.concatenate((data_atlas, add), axis=0)
        num_rows +=1
    
    data = np.reshape(data, (num_rows, num_columns, width, height))
    data_atlas = np.reshape(data_atlas, (num_rows, num_columns, width, height))
    
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    );
    
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img1 = axarr[i, j].imshow(data[i][j], cmap=plt.cm.gray, alpha=0.7,)
                img2 = axarr[i, j].imshow(data_atlas[i][j], cmap=plt.cm.jet, alpha=0.3,)
                axarr[i, j].axis("off");
            else:
                img1 = axarr[j].imshow(data[i][j], cmap=plt.cm.gray, alpha=0.7,)
                img2 = axarr[j].imshow(data_atlas[i][j], cmap=plt.cm.jet, alpha=0.3,)
                axarr[j].axis("off");
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9)
    
    if title:
        f.suptitle(title)
    
    plt.show()