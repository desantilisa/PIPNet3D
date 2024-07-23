import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.auto import tqdm
import monai.transforms as transforms

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def eval_blackbox(
        testloader, 
        model, 
        classes, 
        experiment_folder,
        save=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    targets = []
    
    model.eval()

    with torch.inference_mode():
        for X, y in testloader:
            X = X.to(device)
            y = y.to(device)
            y_logit = model(X)
            y_pred_prob = F.softmax(y_logit, dim=-1) # predicted probability
            y_pred_class = torch.argmax(y_pred_prob, dim=-1) # predicted class label
            predictions.extend(y_pred_class.tolist())
            targets.extend(y.tolist())

    accuracy = accuracy_score(targets, predictions)
    print("Accuracy:", accuracy)

    ConfusionMatrix = confusion_matrix(targets, predictions)
    ConfusionMatrixDisplay.from_predictions(targets, predictions, display_labels = list(classes.keys()), colorbar=False)

    if save:
        np.save(os.path.join(experiment_folder,'ConfusionMatrix'), ConfusionMatrix)
        np.save(os.path.join(experiment_folder,'predictions'), np.array(predictions))
        np.save(os.path.join(experiment_folder,'targets'), np.array(targets))

    report = classification_report(targets, predictions)
    
    return predictions, targets, report
    

    
    
    
    
    
    