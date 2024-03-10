#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:33:55 2023

@author: lisadesanti
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import SimpleITK as sitk
import torch

sys.path.append("3DPIPNet/src/data/")
from make_dataset import get_raw_mri_brains_directories
sys.path.append("/3DPIPNet/src/")
from utils import plot_3d_slices
from utils import get_nested_folders


def atlas_registration(
        img_path, 
        icbm152_path = "/3DPIPNet/data/raw/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii"):
    
    fixed_image = sitk.ReadImage(icbm152_path)
    moving_image = sitk.ReadImage(img_path)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.Similarity3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    moving_resampled = sitk.Resample(
        moving_image, 
        fixed_image, 
        initial_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_image.GetPixelID())
    
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=100, 
        convergenceMinimumValue=1e-6, 
        convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_image, sitk.sitkFloat32))
    
    moving_resampled = sitk.Resample(
        moving_image, 
        fixed_image, 
        final_transform, 
        sitk.sitkLinear, 
        0.0, 
        moving_image.GetPixelID())
    
    return moving_resampled


def skull_stripping(
        img_icbm152_image, 
        icbm152_mask_path = "/3DPIPNet/data/raw/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii"
        ):
    
    """ Apply brain mask to the registered image """
    
    img_icbm152 = sitk.GetArrayFromImage(img_icbm152_image)
    
    icbm152_brain_mask_image = sitk.ReadImage(icbm152_mask_path)
    icbm152_brain_mask = sitk.GetArrayFromImage(icbm152_brain_mask_image)
    
    brain_icbm152 = img_icbm152*icbm152_brain_mask
    
    return brain_icbm152
    
    
def crop_brain(
        brain_icbm152, 
        icbm152_mask_path = "/3DPIPNet/data/raw/mni_icbm152_nlin_sym_09c_nifti/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii",
        margin=3
        ):
    
    """ Remove empty slices from input volume keeping a margin """
    
    icbm152_brain_mask_image = sitk.ReadImage(icbm152_mask_path)
    icbm152_brain_mask = sitk.GetArrayFromImage(icbm152_brain_mask_image)
    
    # Sum brain mask slices along trasverse axis (z axes)
    slice_sums = np.sum(icbm152_brain_mask, axis=(1,2))
    non_empty_slices = np.where(slice_sums > 0)[0]

    # Find the index of the first and last non-zero slice
    idx1 = non_empty_slices[0] # first_non_zero_slice_index
    idx2 = non_empty_slices[-1] # last_non_zero_slice_index
    
    cropped_brain_icbm152 = brain_icbm152[idx1-margin:idx2+margin,:,:]
    
    return cropped_brain_icbm152


def intensity_norm(cropped_brain_icbm152):
    
    """ Normalize voxels' intensity of input volume between 0 and 1 """
    
    img_min = cropped_brain_icbm152.min()
    img_max = cropped_brain_icbm152.max()
    norm_cropped_brain_icbm152 = (cropped_brain_icbm152-img_min)/(img_max-img_min)
    
    return norm_cropped_brain_icbm152
    

def image_preprocessing(
        dataset_path = "/ADNI/ADNI_MRI_NiFTI",
        metadata_path = "/3DPIPNet/data/raw/ADNI1_Screening_1.5T_8_21_2023.csv",
        preprocessed_dataset_path = "/3DPIPNet/data/processed/ADNI_MRI_preprocessed",
        dic_classes = {"CN":0, "AD":1}
        ):
    
    """ Load raw image's directories and apply all the preprocessing steps """
    
    X, y = get_raw_mri_brains_directories(dataset_path, 
                                          metadata_path, 
                                          dic_classes)
    
    size = len(X)
    
    for i, img_path in enumerate(X):
        
        print("Image " + str(i) + " of " + str(size))
        
        img_icbm152_image = atlas_registration(img_path)
        
        brain_icbm152 = skull_stripping(img_icbm152_image)
        
        cropped_brain_icbm152 = crop_brain(brain_icbm152)
        
        norm_cropped_brain_icbm152 = intensity_norm(cropped_brain_icbm152)
        
        all_nested_folders = get_nested_folders(img_path)
        subj = all_nested_folders[5]
        preproc = all_nested_folders[6]
        date = all_nested_folders[7]
        image_data_id = all_nested_folders[8]
        
        img_path = os.path.join(preprocessed_dataset_path, 
                                subj, 
                                preproc, 
                                date,
                                image_data_id)
        
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        np.save(os.path.join(img_path, image_data_id + ".npy"), 
                norm_cropped_brain_icbm152)
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    