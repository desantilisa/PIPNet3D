
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:29:43 2023

@author: 
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



def get_args(
        current_fold: int,      # CURRENT_FOLD: change from 1 to n_fold
        net,                    # Backbone used: "3Dresnet18"
        task_performed,         # "Blackbox" or "PIPNet"
        ) -> argparse.Namespace:
    
    """ 
    Utility functions for handling parsed arguments """
    
    root_folder = "/3DPIPNet"
    dataset_path = os.path.join(root_folder,"data/processed/ADNI_MRI_preprocessed")
    metadata_path = os.path.join(root_folder,"data/raw/ADNI1_Screening_1.5T_8_21_2023.csv")
    
    downscaling = 2
    rows = int(229/downscaling)
    cols = int(193/downscaling)
    slices = int(160/downscaling)
    channels = 3 if net == "3Dresnet18" else 1
    
    dic_classes = {"CN":0, "AD":1} 
    num_classes = len(dic_classes)
    
    if 'PIPNet' in task_performed:
        
        out_shape = 2 if num_classes == 2 else 3
        
        batch_size_pretrain = 12
        batch_size = 12
        epochs_pretrain = 10
        epochs = 60
        optimizer = "Adam"
        lr = 0.05
        lr_block = 0.0005
        lr_net = 0.0005
        weight_decay = 0.0
        num_features = 0
        freeze_epochs = 10
        gamma = 0.1             # LR's decay factor
        step_size = 7           # LR's frequency decay
        
    elif 'Blackbox' in task_performed:
        
        out_shape = 1 if num_classes == 2 else 3
        
        #
        batch_size = 24
        lr = 0.0001
        weight_decay = 0.1
        gamma = 0.1
        step_size = 7
        epochs = 100
        
        # Unused
        batch_size_pretrain = 12
        epochs_pretrain = 10
        optimizer = "Adam"
        lr_block = 0.0005
        lr_net = 0.0005
        num_features = 0
        freeze_epochs = 10
 
    
    classification_task = "Binary_Classification/" if num_classes == 2 else "Multiclass_Classification/"
                          
    n_fold = 5           # Number of fold
    val_split = 0.2
    seed = 42            # seed for reproducible shuffling
    
    experiment_folder = os.path.join(
        root_folder, 
        "runs/results", 
        classification_task, 
        task_performed,
        net, 
        "fold_" + str(current_fold), 
        "Experiment_" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    parser = argparse.ArgumentParser('Train a PIP-Net')    
    parser.add_argument('--task_performed',  type = str, default = task_performed, help = 'String which differentiates between Black-box vs PIPNet training')  
    parser.add_argument('--dataset_path', type = str, default = dataset_path, help = 'Folders path of preprocessed images')
    parser.add_argument('--metadata_path', type = str, default = metadata_path, help = 'Path of .csv metadata file')
    parser.add_argument('--downscaling', type = int, default = downscaling,  help = 'Subsampling factor')
    parser.add_argument('--rows', type = int, default = rows, help = 'Number of rows in input image')
    parser.add_argument('--cols', type = int, default = cols, help = 'Number of columns in input image')
    parser.add_argument(' --slices', type = int, default = slices, help = 'Number of slices in input image')
    parser.add_argument('--channels', type = int, default = channels, help = 'N° of channel of the input volume passed to the network')
    parser.add_argument( '--img_shape', type = tuple, default = (slices, rows, cols), help = 'Shape of the input volume passed to the network')
    parser.add_argument('--dic_classes', type = dict, default = dic_classes, help = 'Dictionary "labels": class_id')
    parser.add_argument('--num_classes', type = int,  default = num_classes,  help = 'Subsampling factor')
    parser.add_argument('--out_shape', type = int, default = out_shape, help = 'Subsampling factor')
    parser.add_argument('--net', type = str, default = net, help = 'Base network used as backbone of PIP-Net. ')
    parser.add_argument('--freeze_extractor', type = bool, default = False,  help = '')
    parser.add_argument('--current_fold',  type = int, default = current_fold, help = 'N° of the current fold used ad test set')
    parser.add_argument('--n_fold', type = int, default = n_fold, help = 'N° of fold in the k-fold cross validation')
    parser.add_argument('--val_split', type = float, default = val_split, help = 'Percentage of validation set')
    parser.add_argument('--seed', type = int, default = seed, help = 'Random seed. Note that there will still be differences between runs due to nondeterminism. See https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--experiment_folder', type = str,  default = experiment_folder, help = 'String which differentiates between Black-box vs PIPNet  training')
    parser.add_argument('--batch_size', type = int, default = batch_size, help = 'Batch size when training the model using minibatch gradient descent. Batch size is multiplied with number of available GPUs')
    parser.add_argument('--batch_size_pretrain', type = int, default = batch_size_pretrain, help = 'Batch size when pretraining the prototypes (first training  stage)')
    parser.add_argument('--epochs', type = int, default = epochs, help = 'The number of epochs PIP-Net should be trained (second  training stage)')
    parser.add_argument('--epochs_pretrain', type = int, default = epochs_pretrain,  help = 'Number of epochs to pre-train the prototypes (first training  stage). Recommended to train at least until the align loss < 1')
    parser.add_argument('--optimizer', type = str, default = optimizer, help = 'The optimizer that should be used when training PIP-Net')
    parser.add_argument('--lr', type = float, default = lr, help = 'The optimizer learning rate for training the weights from  prototypes to classes')
    parser.add_argument('--lr_block',  type = float, default = lr_block, help = 'The optimizer learning rate for training the last conv layers  of the backbone')
    parser.add_argument('--lr_net', type = float, default = lr_net, help = 'The optimizer learning rate for the backbone. Usually similar  as lr_block.')        
    parser.add_argument('--weight_decay', type = float, default = weight_decay, help = 'Weight decay used in the optimizer')
    parser.add_argument('--num_features', type = int, default = num_features, help = 'Number of prototypes. When zero (default) the number of  prototypes is the number of output channels of backbone. If this  value is set, then a 1x1 conv layer will be added. Recommended to  keep 0, but can be increased when number of classes > num output  channels in backbone.')
    parser.add_argument('--freeze_epochs',  type = int, default = freeze_epochs, help = 'Number of epochs where pretrained features_net will be frozen while training classification layer (and last layer(s) of backbone)')
    parser.add_argument('--gamma', type = float, default = gamma, help = 'Learning rate decay factor')
    parser.add_argument('--step_size', type = int, default = step_size, help = 'Learning rate frequency decay')
    parser.add_argument('--disable_cuda', action = 'store_true',  help = 'Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',  type = str, default = experiment_folder, help = 'The directory in which train progress should be logged')
    parser.add_argument('--state_dict_dir_net', type = str, default = '',  help = 'The directory containing a state dict with a pretrained  PIP-Net. E.g., ./runs/run_pipnet/checkpoints/net_pretrained')
    parser.add_argument( '--dir_for_saving_images', type = str, default = 'visualization_results', help = 'Directoy for saving the prototypes and explanations')
    parser.add_argument('--disable_pretrained', action = 'store_true',  help = 'When set, the backbone network is initialized with random weights instead of being pretrained on another dataset).')
    parser.add_argument('--weighted_loss', action = 'store_true', help = 'Flag that weights the loss based on the class balance of the  dataset. Recommended to use when data is imbalanced. ')
    parser.add_argument('--gpu_ids', type = str, default = '', help = 'ID of gpu. Can be separated with comma')
    parser.add_argument('--num_workers', type = int, default = 2, help = 'Num workers in dataloaders.')
    parser.add_argument('--bias', default = False, action = 'store_true', help = 'Flag that indicates whether to include a trainable bias in  the linear classification layer.')
    parser.add_argument('--extra_test_image_folder', type = str, default = './experiments', help = 'Folder with images that PIP-Net will predict and explain,  that are not in the training or test set. E.g. images with 2  objects or OOD image. Images should be in subfolder. E.g. images  in ./experiments/images/, and argument --./experiments')
    args = parser.parse_args()
    
    if 'Train' in task_performed:
    
        if len(args.log_dir.split('/'))>2:
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)

    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    
    """
    Save the arguments in the specified directory as
        - a text file called 'args.txt'
        - a pickle file called 'args.pickle'
    :param args: The arguments to be saved
    :param directory_path: The path to the directory where the arguments should 
    be saved """
    
    # If the specified directory does not exists, create it
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):    # Add quotation marks to indicate that 
                                        # the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)     
    

class Log:

    """ Object for managing the log directory """

    def __init__(self, log_dir: str):  # Store log in log_dir

        self._log_dir = log_dir
        self._logs = dict()

        # Ensure the directories exist
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.metadata_dir):
            os.mkdir(self.metadata_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self._log_dir + '/metadata'

    def log_message(self, msg: str):
        """
        Write a message to the log file
        :param msg: the message string to be written to the log file
        """
        if not os.path.isfile(self.log_dir + '/log.txt'):
            open(self.log_dir + '/log.txt', 'w').close() # make log file empty if it already exists
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write(msg+"\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        """
        Create a csv for logging information
        :param log_name: The name of the log. The log filename will be 
                         <log_name>.csv.
        :param key_name: The name of the attribute that is used as key (e.g. 
                         epoch number)
        :param value_names: The names of the attributes that are logged
        """
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        # Add to existing logs
        self._logs[log_name] = (key_name, value_names)
        # Create log file. Create columns
        with open(self.log_dir + f'/{log_name}.csv', 'w') as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        """
        Log values in an existent log file
        :param log_name: The name of the log file
        :param key: The key attribute for logging these values
        :param values: value attributes that will be stored in the log
        """
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # Write a new line with the given values
        with open(self.log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')

    def log_args(self, args: argparse.Namespace):
        save_args(args, self._log_dir)


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
                img = axarr[i, j].imshow(data[i][j], 
                                         cmap=cmap,
                                         vmin=data_min, 
                                         vmax=data_max)
                axarr[i, j].axis("off");
            else:
                img = axarr[j].imshow(data[i][j], 
                                      cmap=cmap,
                                      vmin=data_min, 
                                      vmax=data_max)
                axarr[j].axis("off");

    plt.subplots_adjust(wspace=0,
                        hspace=0,
                        left=0,
                        right = 0.9,
                        bottom=0,
                        top=0.9)
    
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
                img = axarr[i, j].imshow(data[i][j], 
                                         vmin=data_min, 
                                         vmax=data_max)
                axarr[i, j].axis("off");
            else:
                img = axarr[j].imshow(data[i][j], 
                                      vmin=data_min, 
                                      vmax=data_max)
                axarr[j].axis("off");

    plt.subplots_adjust(wspace=0,
                        hspace=0,
                        left=0,
                        right = 0.9,
                        bottom=0,
                        top=0.9)
    
    if title:
        if bottom:
            f.suptitle(title, 
                       fontsize="large",
                       y=0., 
                       va="top",
                       color="gray")
        else:
            f.suptitle(title)
            
    if save_path:
        f.savefig(save_path, bbox_inches='tight')
        
    plt.show()


def plot_atlas_overlay(data, data_atlas, num_columns=10, title=False,):
    
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
                img1 = axarr[i, j].imshow(data[i][j], 
                                         cmap=plt.cm.gray, 
                                         alpha=0.7,)
                img2 = axarr[i, j].imshow(data_atlas[i][j], 
                                         cmap=plt.cm.jet, 
                                         alpha=0.3,)
                axarr[i, j].axis("off");
            else:
                img1 = axarr[j].imshow(data[i][j], 
                                      cmap=plt.cm.gray, 
                                      alpha=0.7,)
                img2 = axarr[j].imshow(data_atlas[i][j], 
                                      cmap=plt.cm.jet, 
                                      alpha=0.3,)
                axarr[j].axis("off");
    
    plt.subplots_adjust(wspace=0,
                        hspace=0,
                        left=0,
                        right = 0.9,
                        bottom=0,
                        top=0.9)
    
    if title:
        f.suptitle(title)
    
    plt.show()
    
    
def generate_rgb_array(n):
    
    # Generate a list of n distinct colors using matplotlib's 'tab10' colormap
    cmap = plt.get_cmap('tab10')
    color_list = [cmap(i)[:3] for i in np.linspace(0, 1, n)]

    # Convert the list to a NumPy array
    rgb_array = np.array(color_list)

    return rgb_array   
    
    
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


def init_weights_xavier(m):
    if type(m) == torch.nn.Conv3d:
        torch.nn.init.xavier_uniform_(
            m.weight, 
            gain = torch.nn.init.calculate_gain('sigmoid'))
  

def get_optimizer_nn(
        net, 
        args: argparse.Namespace
        ) -> torch.optim.Optimizer:
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # create parameter groups
    params_to_freeze = []
    params_to_train = []
    params_backbone = []
    
    # set up optimizer
    if '3Dresnet18' or '3Duxnet' in args.net:
        print("Network is ", args.net, flush = True)
        # Train all the backbone
        for name, param in net.module._net.named_parameters():
            params_to_train.append(param)
    else:
        print("Network not implemented", flush = True)     
    
    classification_weight = []
    classification_bias = []
    for name, param in net.module._classification.named_parameters():
        if 'weight' in name:
            classification_weight.append(param)
        elif 'multiplier' in name:
            param.requires_grad = False
        else:
            if args.bias:
                classification_bias.append(param)
    
    paramlist_net = [
            {"params": params_backbone, 
             "lr": args.lr_net, 
             "weight_decay_rate": args.weight_decay},
            {"params": params_to_freeze, 
             "lr": args.lr_block, 
             "weight_decay_rate": args.weight_decay},
            {"params": params_to_train, 
             "lr": args.lr_block, 
             "weight_decay_rate": args.weight_decay},
            {"params": net.module._add_on.parameters(), 
             "lr": args.lr_block*10., 
             "weight_decay_rate": args.weight_decay}]
            
    paramlist_classifier = [
            {"params": classification_weight, 
             "lr": args.lr, 
             "weight_decay_rate": args.weight_decay},
            {"params": classification_bias, 
             "lr": args.lr, 
             "weight_decay_rate": 0},]
          
    if args.optimizer == 'Adam':
        optimizer_net = torch.optim.AdamW(
            paramlist_net,
            lr = args.lr,
            weight_decay = args.weight_decay)
        optimizer_classifier = torch.optim.AdamW(
            paramlist_classifier,
            lr = args.lr,
            weight_decay = args.weight_decay)
        return optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone
    
    else:
        raise ValueError("this optimizer type is not implemented")


def topk_accuracy(output, target, topk=[1,]):
    
    """
    Computes the accuracy over the k top predictions for the specified values 
    of k """
    
    with torch.no_grad():
        topk2 = [x for x in topk if x <= output.shape[1]] # ensures that k is 
                                                          # not larger than 
                                                          # number of classes
        maxk = max(topk2)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            if k in topk2:
                correct_k = correct[:k].reshape(-1).float()
                res.append(correct_k)
            else:
                res.append(torch.zeros_like(target))
        return res


def get_patch_size(args):
    
    patchsize = 32
    skip_z = round((args.img_shape[0] - patchsize) / (args.dshape-1))
    skip_y = round((args.img_shape[1] - patchsize) / (args.hshape-1))
    skip_x = round((args.img_shape[2] - patchsize) / (args.wshape-1))
    
    return patchsize, skip_z, skip_y, skip_x







