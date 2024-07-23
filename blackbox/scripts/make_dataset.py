#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:24:11 2023

@author: lisadesanti
"""

import argparse
import os
import math
import numpy as np
import random
import pandas 
import pydicom
import SimpleITK as sitk

import torch
from torch import Tensor
import torch.optim
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def get_raw_mri_brains_directories(
        dataset_path = "/home/lisadesanti/DeepLearning/ADNI/ADNI_DATASET/ADNI_MRI_NiFTI",
        metadata_path = "/home/lisadesanti/DeepLearning/ADNI/ADNI_DATASET/ADNI1_Screening_1.5T_8_21_2023.csv",
        dic_classes = {"CN":0, "MCI":1, "AD":2}):
    
    """ Get directories and labels of sMRI images
    
    Args:
        
        - dataset_path: Path of folder containing input images
        - metadata_path: Path of folder containing metadata (including image labels) 
        - dic_classes: Dictionary, "class_name":label of classes considered
    
    Returns: 
        
        - ndarray of data directories
        - ndarray of image labels
        - dict containing: list of subjects, # subjects, # CN/MCI/AD 
            respectively in training, validation and test set

    """
    
    metadata_path = pandas.read_csv(metadata_path)
    subjs_id = os.listdir(dataset_path)
    img_directiories_dic = []
    labels = []
    
    for subj_id in subjs_id:
        
        subj_path = os.path.join(dataset_path, subj_id)
        process_types = os.listdir(subj_path)
        
        for process_type in process_types:
            
            process_path = os.path.join(subj_path, process_type)
            acq_dates = os.listdir(process_path)
            
            for acq_date in acq_dates:
                
                acq_path = os.path.join(process_path, acq_date)
                img_id = os.listdir(acq_path)[0]
                img_folder = os.path.join(acq_path, img_id)
                img_file = os.listdir(img_folder)[0]
                img_directory = os.path.join(img_folder, img_file)
                # Find image label (cognitive decline level) in metadata file
                label = metadata_path.loc[metadata_path['Image Data ID'] == img_id]['Group']
                label = label.to_numpy()[0]
                
                img_directory_dic = {"ROOT":dataset_path, "LABEL":label, "SUBJ":subj_id, "PREPROC":process_type, "DATE":acq_date, "EXAM_ID":img_id, "FILENAME":img_file}
                
                if label in dic_classes.keys():
		    # Save only image directory according to the classification task performed
                    img_directiories_dic.append(img_directory_dic)
                else:
                    continue
    
    # Dataframe of ADNI directories
    directory_dataframe = pandas.DataFrame(img_directiories_dic)
    img_num = len(img_directiories_dic)

    X = np.array(directory_dataframe['ROOT']) + \
        np.array(['/']*img_num) + \
        np.array(directory_dataframe['SUBJ']) + \
        np.array(['/']*img_num) + \
        np.array(directory_dataframe['PREPROC']) + \
        np.array(['/']*img_num) + \
        np.array(directory_dataframe['DATE']) + \
        np.array(['/']*img_num) + \
        np.array(directory_dataframe['EXAM_ID']) + \
        np.array(['/']*img_num) + \
        np.array(directory_dataframe['FILENAME'])
    y = np.array(directory_dataframe["LABEL"])
    y = np.array([dic_classes[yi] for yi in y])
    
    return X, y


def get_mri_brains_paths(
        dataset_path = "/home/lisadesanti/DeepLearning/ADNI/ADNI_DATASET/ADNI_MRI_preprocessed",
        metadata_path = "/home/lisadesanti/DeepLearning/ADNI/ADNI_DATASET/ADNI1_Screening_1.5T_8_21_2023.csv",
        dic_classes = {"CN":0, "MCI":1, "AD":2},
        set_type = 'train',
        shuffle = True,
        n_fold = 5, 
        current_fold = 1,
        test_split = 0.2,
        seed = 42):
    
    """ Get a list with directories and labels of the preprocessed sMRI dataset 
    of the Training/Validation/Test set splitted performing 5-fold 
    cross-validation
    
    Args:
        
        - dataset_path: Path of folder containing input images
        - metadata_path: Path of folder containing metadata (including image labels) 
        - dic_classes: Dictionary of classes considered, "class_name":label
        - set_type: str used to identify Training/Validation/Test set:
            set_type \in {"train", "val", "test"}
        - current_fold: Current fold 
        - n_fold: n° fold in kfold validation
        - test_split: % of split used
    
    Returns: 
        
        - ndarray of Training/Validation/Test/All data directories
        - ndarray of image labels
        - dict containing: list of subjects, # subjects, # CN/MCI/AD 
            respectively in training, validation and test set
   
    """
    
    metadata_paths = pandas.read_csv(metadata_path)
    subjs_id = os.listdir(dataset_path)
    img_directiories_dic = []
    labels = []
    
    for subj_id in subjs_id:
        
        subj_path = os.path.join(dataset_path, subj_id)
        process_types = os.listdir(subj_path)
        
        for process_type in process_types:
            
            process_path = os.path.join(subj_path, process_type)
            acq_dates = os.listdir(process_path)
            
            for acq_date in acq_dates:
                
                acq_path = os.path.join(process_path, acq_date)
                img_id = os.listdir(acq_path)[0]
                img_folder = os.path.join(acq_path, img_id)
                img_file = os.listdir(img_folder)[0]
                img_directory = os.path.join(img_folder, img_file)
                
                # Find image label (cognitive decline level) in metadata file
                label = metadata_paths.loc[metadata_paths['Image Data ID'] == img_id]['Group']
                label = label.to_numpy()[0]
                # Find patient age in metadata file
                age = metadata_paths.loc[metadata_paths['Image Data ID'] == img_id]['Age']
                age = age.to_numpy()[0]
                
                img_directory_dic = {"ROOT":dataset_path, "LABEL":label, "SUBJ":subj_id, "PREPROC":process_type, "DATE":acq_date, "EXAM_ID":img_id, "FILENAME":img_file, "AGE":age}
                
                if label in dic_classes.keys():
		            # Save only image directory according to the classification 
                    # task performed
                    img_directiories_dic.append(img_directory_dic)
                else:
                    continue
    
    # Dataframe of ADNI directories
    directory_dataframe = pandas.DataFrame(img_directiories_dic)
    all_subj = list(set(directory_dataframe["SUBJ"]))
    labels = directory_dataframe["LABEL"]
    ages = directory_dataframe["AGE"]
    img_num = len(img_directiories_dic)
    
    # Split Dataset into Training(+ Valid) and Test set 
    # Shuffle (reproducible) and select the last 20% of the dataset 
    X_train_val_df, X_test_df, y_train_val, y_test = train_test_split(
        directory_dataframe, 
        labels, 
        test_size = test_split, 
        shuffle = True, 
        random_state = seed, 
        stratify = labels
        ); # with shuffle False stratify is not support
    
    # Check to not have data (exams) from the same subjects both in the 
    # training and validation sets
    subj_train = np.array(X_train_val_df["SUBJ"])
    subj_test = np.array(X_test_df["SUBJ"])
    dup_subjects = np.intersect1d(subj_train, subj_test)

    # If a subjects has data in both sets move data to the training set
    for dup_subj in dup_subjects:

        dup_subj_test = X_test_df.loc[X_test_df["SUBJ"]==dup_subj]
        id_dup_subj_test = np.array(dup_subj_test.index)
        to_train = X_test_df.loc[id_dup_subj_test]

        # Test set (without duplicated subjects)
        X_test_df = X_test_df.drop(id_dup_subj_test)
        X_test_df = X_test_df.sort_values("SUBJ")
        y_test = X_test_df["LABEL"]
        
        # Training+Validation set (without duplicated subjects)
        X_train_val_df = pandas.concat([X_train_val_df, to_train], ignore_index=True)
        X_train_val_df = X_train_val_df.sort_values("SUBJ")
        y_train_val = X_train_val_df["LABEL"]
        
    # Perform k-fold crossvalidation on the Training + Validation
    
    # Create a new index
    new_index = range(0, 0 + len(X_train_val_df))
    # Reindex the DataFrame
    X_train_val_df.index = new_index
    y_train_val.index = new_index
    skf = StratifiedKFold(n_splits=n_fold, random_state=None, shuffle=False)
    
    kfold_generator = skf.split(X_train_val_df, y_train_val)

    for i in range(current_fold):
        
        # Split into Training and Validation set
        train_index, val_index = next(kfold_generator)
        X_train_df  = X_train_val_df.loc[train_index]
        X_val_df = X_train_val_df.loc[val_index]
        y_train = y_train_val[train_index]
        y_val = y_train_val[val_index]
   
        # Check to not have data (exams) of the same subjects both in the 
        # training and test sets
        subj_train = np.array(X_train_df["SUBJ"])
        subj_val = np.array(X_val_df["SUBJ"])
        dup_subjects = np.intersect1d(subj_train, subj_val)
   
        # If a subjects has data in both sets move data to the training set
        # (this is an arbitrary choice)
        for dup_subj in dup_subjects:
   
            dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj]
            id_dup_subj_val = np.array(dup_subj_val.index)
            to_train = X_val_df.loc[id_dup_subj_val]
   
            # Validation set (without duplicated subjects)
            X_val_df = X_val_df.drop(id_dup_subj_val)
            X_val_df = X_val_df.sort_values("SUBJ")
            y_val = X_val_df["LABEL"]
            
            # Training set (without duplicated subjects)
            X_train_df = pandas.concat([X_train_df, to_train], ignore_index=True)
            X_train_df = X_train_df.sort_values("SUBJ")
            y_train = X_train_df["LABEL"]

    # Check to not have data (exams) from the same subjects both in the 
    # training and validation sets
    subj_train = np.array(X_train_df["SUBJ"])
    subj_val = np.array(X_val_df["SUBJ"])
    dup_subjects = np.intersect1d(subj_train, subj_val)

    # If a subjects has data in both sets move data to the training set
    # (this is an arbitrary choice)
    for dup_subj in dup_subjects:

        dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj]
        id_dup_subj_val = np.array(dup_subj_val.index)
        to_train = X_val_df.loc[id_dup_subj_val]

        # Vslidation set (without duplicated subjects)
        X_val_df = X_val_df.drop(id_dup_subj_val)
        X_val_df = X_val_df.sort_values("SUBJ")
        y_val = X_val_df["LABEL"]
        
        # Training+Validation set (without duplicated subjects)
        X_train_df = pandas.concat([X_train_df, to_train], ignore_index=True)
        X_train_df = X_train_df.sort_values("SUBJ")
        y_train = X_train_df["LABEL"]
    
    subj_train = np.array(X_train_df["SUBJ"])
    n_train = len(X_train_df)
    train_cn = y_train.tolist().count('CN')
    train_mci = y_train.tolist().count('MCI')
    train_ad = y_train.tolist().count('AD')
    
    subj_val = np.array(X_val_df["SUBJ"])
    n_val = len(X_val_df)
    val_cn = y_val.tolist().count('CN')
    val_mci = y_val.tolist().count('MCI')
    val_ad = y_val.tolist().count('AD')
    
    subj_test = np.array(X_test_df["SUBJ"])
    n_test = len(X_test_df)
    test_cn = y_test.tolist().count('CN')
    test_mci = y_test.tolist().count('MCI')
    test_ad = y_test.tolist().count('AD')
    
    dataset_info = {"train_subj":subj_train.tolist(),
                    "n_train":n_train,
                    "train_cn":train_cn,
                    "train_mci":train_mci,
                    "train_ad":train_ad,
                    "val_subj":subj_val.tolist(),
                    "n_val":n_val,
                    "val_cn":val_cn,
                    "val_mci":val_mci,
                    "val_ad":val_ad,
                    "test_subj":subj_test.tolist(),
                    "n_test":n_test,
                    "test_cn":test_cn,
                    "test_mci":test_mci,
                    "test_ad":test_ad}
    
    dup_subjects_train_val = np.intersect1d(subj_train, subj_val)
    dup_subjects_train_test = np.intersect1d(subj_train, subj_test)
    dup_subjects_val_test = np.intersect1d(subj_val, subj_test)
    
    # Check data leackage issue
    if len(dup_subjects_train_val) or len(dup_subjects_train_test) or len(dup_subjects_val_test):
        print('Data Leackage occurred!! ')
        return
    
    X = np.array(directory_dataframe['ROOT']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['SUBJ']) +  np.array(['/']*img_num) + \
        np.array(directory_dataframe['PREPROC']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['DATE']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['EXAM_ID']) + np.array(['/']*img_num) + \
        np.array(directory_dataframe['FILENAME'])
    x_age = np.array(ages)
    y = np.array(labels)
    
    
    X_train = np.array(X_train_df['ROOT']) + np.array(['/']*n_train) + \
              np.array(X_train_df['SUBJ']) + np.array(['/']*n_train) + \
              np.array(X_train_df['PREPROC']) + np.array(['/']*n_train) + \
              np.array(X_train_df['DATE']) + np.array(['/']*n_train) + \
              np.array(X_train_df['EXAM_ID']) + np.array(['/']*n_train) + \
              np.array(X_train_df['FILENAME'])
    x_age_train = np.array(X_train_df['AGE'])
    y_train = np.array(y_train)
    y_train = np.array([dic_classes[yi] for yi in y_train])
    
    X_val = np.array(X_val_df['ROOT']) + np.array(['/']*n_val) + \
            np.array(X_val_df['SUBJ']) + np.array(['/']*n_val) + \
            np.array(X_val_df['PREPROC']) + np.array(['/']*n_val) + \
            np.array(X_val_df['DATE']) +  np.array(['/']*n_val) + \
            np.array(X_val_df['EXAM_ID']) + np.array(['/']*n_val) + \
            np.array(X_val_df['FILENAME'])
    x_age_val = np.array(X_val_df['AGE'])
    y_val = np.array(y_val)
    y_val = np.array([dic_classes[yi] for yi in y_val])
    
    X_test = np.array(X_test_df['ROOT']) + np.array(['/']*n_test) + \
             np.array(X_test_df['SUBJ']) + np.array(['/']*n_test) + \
             np.array(X_test_df['PREPROC']) + np.array(['/']*n_test) + \
             np.array(X_test_df['DATE']) + np.array(['/']*n_test) + \
             np.array(X_test_df['EXAM_ID']) + np.array(['/']*n_test) + \
             np.array(X_test_df['FILENAME'])
    x_age_test = np.array(X_test_df['AGE'])
    y_test = np.array(y_test)
    y_test = np.array([dic_classes[yi] for yi in y_test])

    # Data shuffling 
    if shuffle:
        
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_train)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_train = X_train[shuffled_index]
        x_age_train = x_age_train[shuffled_index]
        y_train = y_train[shuffled_index]
    
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_val)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_val = X_val[shuffled_index]                      
        x_age_val = x_age_val[shuffled_index]
        y_val = y_val[shuffled_index]
        
        rng = np.random.default_rng(seed)
        shuffled_index = np.arange(n_test)
        rng.shuffle(shuffled_index)
        # Shuffled dataset
        X_test = X_test[shuffled_index]                    
        x_age_test = x_age_test[shuffled_index]
        y_test = y_test[shuffled_index]
    
    if set_type == 'train':
        return X_train, x_age_train, y_train, dataset_info
    elif set_type == 'val':
        return X_val, x_age_val, y_val, dataset_info
    elif set_type == 'test':
        return X_test, x_age_test, y_test, dataset_info
    else:
        return X, x_age, y, dataset_info
    


class BrainDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_type: str,
            dic_classes: dict,
            transform = None,
            set_type = 'train',
            n_fold = 5,
            current_fold = 1,
            test_split = 0.2,
            seed = 42):
        
        self.set_type = set_type      # tranformation's type
        
        dataset = get_mri_brains_paths(
            dic_classes = dic_classes,
            set_type = self.set_type,
            n_fold = n_fold,
            current_fold = current_fold,
            test_split = test_split,
            seed = seed) 
        
        self.img_dir = dataset[0]      # ndarray with images directories
        self.img_age = dataset[1]      # ndarray with patient age
        self.img_labels = dataset[2]   # ndarray with images labels
        self.dataset_info = dataset[3] # dictionary with dataset info
        self.transform = transform[set_type] # images transformation
        self.classes = list(dic_classes.keys())
        self.class_to_idx = dic_classes
        
        
    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        
        img_path = self.img_dir[idx]
        label = self.img_labels[idx]
        
        image = np.load(img_path).astype(np.float32)
        volume = torch.tensor(image) # torch.Size([slices, rows, cols])
        volume = torch.unsqueeze(volume, 0) # add channel dimension
        volume = volume.float()
        
        if self.transform:
            volume = self.transform(volume)
            img_min = volume.min()
            img_max = volume.max()
            volume = (volume-img_min)/(img_max-img_min)

        return volume, label

    


