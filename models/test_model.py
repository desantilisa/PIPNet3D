import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import monai.transforms as transforms
import SimpleITK as sitk
from itertools import combinations

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from scipy.stats import entropy

sys.path.append("/3DPIPNet/src/")
from utils import Log
from utils import topk_accuracy
from utils import get_patch_size, plot_3d_slices, plot_rgb_slices, generate_rgb_array
sys.path.append("/3DPIPNet/src/features")
from build_features import crop_brain
sys.path.append("/3DPIPNet/src/visualization")
from vis_pipnet import get_img_coordinates, plot_local_explanation



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
            
            if y_logit.shape[1] == 1:
                y_logit = y_logit.squeeze(dim=1) # logit
                y_pred_prob = torch.sigmoid(y_logit) # predicted probability
                y_pred_class = torch.round(y_pred_prob) # label
                y = y.float()
            else:
                y_pred_prob = torch.softmax(y_logit, dim=1) # predicted probability
                y_pred_class = torch.argmax(y_pred_prob, dim=1)
            
            predictions.extend(y_pred_class.tolist())
            targets.extend(y.tolist())

    accuracy = accuracy_score(targets, predictions)
    print("Accuracy:", accuracy)

    ConfusionMatrix = confusion_matrix(targets, predictions)
    ConfusionMatrixDisplay.from_predictions(targets,
                                            predictions,
                                            display_labels = list(classes.keys()),
                                            colorbar=False)
    
    if save:
        
        np.save(os.path.join(experiment_folder,'ConfusionMatrix'), 
                ConfusionMatrix)
        np.save(os.path.join(experiment_folder,'predictions'), 
                np.array(predictions))
        np.save(os.path.join(experiment_folder,'targets'), np.array(targets))

    report = classification_report(targets, predictions)
    
    return predictions, targets, report
    

@torch.no_grad()
def eval_pipnet(
        net,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()
    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((net.module._num_classes, net.module._num_classes), 
                  dtype = int)

    global_top1acc = 0.
    global_top5acc = 0.
    global_sim_anz = 0.
    global_anz = 0.
    local_size_total = 0.
    y_trues = []
    y_preds = []
    y_preds_classes = []
    abstained = 0
    
    test_iter = iter(test_loader)
    
    # Iterate through the test set
    for i, (xs, ys) in enumerate(test_iter):
        
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            
            # net.module._classification.weight.copy_(torch.clamp(
            #     net.module._classification.weight.data - 1e-3, min = 0.))
            
            # Use the model to classify this batch of input data
            _, pooled, out = net(xs, inference = True)
            
            max_out_score, ys_pred = torch.max(out, dim=1) # max, max_idx
            
            ys_pred_scores = torch.amax(F.softmax((torch.log1p(
                out**net.module._classification.normalization_multiplier)),
                dim = 1), dim = 1) # class confidence scores
            
            abstained += (max_out_score.shape[0] - \
                          torch.count_nonzero(max_out_score))
                
            repeated_weight = net.module._classification.weight.unsqueeze(
                1).repeat(1, pooled.shape[0], 1)
            
            sim_scores_anz = torch.count_nonzero(torch.gt(torch.abs(
                pooled*repeated_weight), 1e-3).float(), dim = 2).float()
            
            local_size = torch.count_nonzero(torch.gt(
                torch.relu((pooled*repeated_weight) - 1e-3).sum(dim = 1), 
                0.).float(), dim = 1).float()
            
            local_size_total += local_size.sum().item()

            correct_class_sim_scores_anz = torch.diagonal(torch.index_select(
                sim_scores_anz, dim = 0, index = ys_pred), 0)
            
            global_sim_anz += correct_class_sim_scores_anz.sum().item()
            
            almost_nz = torch.count_nonzero(torch.gt(
                torch.abs(pooled), 1e-3).float(), dim = 1).float()
            
            global_anz += almost_nz.sum().item()
            
            # Update the confusion matrix
            cm_batch = np.zeros((net.module._num_classes, 
                                 net.module._num_classes), dtype = int)
            
            for y_pred, y_true in zip(ys_pred, ys):
                
                cm[y_true][y_pred] += 1
                cm_batch[y_true][y_pred] += 1
                
            acc = acc_from_cm(cm_batch)   

            (top1accs, top5accs) = topk_accuracy(out, ys, topk=[1,5])
            
            global_top1acc += torch.sum(top1accs).item()
            global_top5acc += torch.sum(top5accs).item()
            y_preds += ys_pred_scores.detach().tolist()     # predicted class' confidence scores
            y_trues += ys.detach().tolist()
            y_preds_classes += ys_pred.detach().tolist()    # predicted classes
        
        del out
        del pooled
        del ys_pred
        
    print("PIP-Net abstained from a decision for", 
          abstained.item(), 
          "images", 
          flush = True)     
       
    info['num non-zero prototypes'] = torch.gt(
        net.module._classification.weight, 1e-3).any(dim = 0).sum().item()
    
    print(
        "sparsity ratio: ",
        (torch.numel(net.module._classification.weight) - torch.count_nonzero(
            torch.nn.functional.relu(net.module._classification.weight-1e-3)
            ).item()) / torch.numel(net.module._classification.weight), 
        flush = True)
    
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    info['top1_accuracy'] = global_top1acc/len(test_loader.dataset)
    info['top5_accuracy'] = global_top5acc/len(test_loader.dataset)
    info['almost_sim_nonzeros'] = global_sim_anz/len(test_loader.dataset)
    info['local_size_all_classes'] = local_size_total/len(test_loader.dataset)
    info['almost_nonzeros'] = global_anz/len(test_loader.dataset)
    
    report = classification_report(y_trues, y_preds_classes)
    print("F1", f1_score(y_trues, y_preds_classes), flush = True)
    print(report, flush = True)
    
    if net.module._num_classes == 2:
        
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        print("TP: ", tp, "FN: ",fn, "FP:", fp, "TN:", tn, flush=True)
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        
        print("\n Epoch",epoch, flush=True)
        print("Confusion matrix: ", cm, flush=True)
        
        try:
            for classname, classidx in test_loader.dataset.class_to_idx.items(): 
                if classidx == 0:
                    print("Accuracy positive class (", 
                          classname, 
                          classidx,
                          ") (TPR, Sensitivity):", 
                          tp/(tp+fn))
                elif classidx == 1:
                    print("Accuracy negative class (", 
                          classname, 
                          classidx,
                          ") (TNR, Specificity):", 
                          tn/(tn+fp))
                    
        except ValueError:
            pass
        
        print("Balanced accuracy: ", 
              balanced_accuracy_score(y_trues, y_preds_classes),
              flush = True)
        print("Sensitivity: ", 
              sensitivity, 
              "Specificity: ", 
              specificity,
              flush = True)
        info['top5_accuracy'] = f1_score(y_trues, y_preds_classes)
        
        try:
            print("AUC macro: ", 
                  roc_auc_score(y_trues, y_preds, average='macro'), 
                  flush = True)
            print("AUC weighted: ", 
                  roc_auc_score(y_trues, y_preds, average='weighted'), 
                  flush  =True)
        
        except ValueError:
            pass
        
    else:
        info['top5_accuracy'] = global_top5acc/len(test_loader.dataset) 

    return info

    
def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total


@torch.no_grad()                    
def get_local_explanations(
        net, 
        projectloader, 
        device,
        args: argparse.Namespace,
        plot = False):
    """
    Compute the local explanations for all the images in the dataloader.
    Get the coordinates of the image patches of the relevant prototypes 
    detected with similarity > 0.1 in each prediction
    
    Return: 
        - local_explanations: list of the local explanations of projectloader.
          Every local explanation is a dict containing all the detected 
          prototypes in the input passed, where:
              - key: int, index which identity the detected prototype
              - value: tuple containing:
                  - (dmin,dmax,hmin,hmax,wmin,wmax): tuple of the coordinates
                    in input image of the detected prototype
                  - simweight: contribution of the detected prototype to the
                    class predicted
        - y_preds
        - y_trues
        
    """
    
    print("Detect prototypes in predictions...", flush = True)

    local_explanations = []
    y_preds = []
    y_trues = []
    
    patchsize, skip_z, skip_y, skip_x = get_patch_size(args)

    imgs = [(img, label) for img, label in zip(
        projectloader.dataset.img_dir, projectloader.dataset.img_labels)]
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight

    img_iter = enumerate(iter(projectloader))
    
    for k, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        
        xs, ys = xs.to(device), ys.to(device)
        img = imgs[k][0]
        
        local_explanation = dict() # dict of all detected prototypes
        
        with torch.no_grad():
            
            # NOTE: inference=True -> ignore prototypes with similarity < 0.1
            # (elements of pooled < 0.1 are set to 0)
            softmaxes, pooled, out = net(xs, inference=True) # softmaxes: (1,num_ps,d,h,w)
                                                             # pooled:    (bs,num_ps), 
                                                             # out:       (bs,num_classes)
            
            sorted_out, sorted_out_indices = torch.sort(
                out.squeeze(0), descending=True) # (num_classes)
            
            max_out_score, ys_pred = torch.max(out, dim=1) # max, max_idx
            
            y_preds.append(ys_pred.item())
            y_trues.append(ys.item())
            
            for pred_class_idx in sorted_out_indices:
                
                sorted_pooled, sorted_pooled_indices = torch.sort(
                    pooled.squeeze(0), descending=True) # (num_ps) -> prototypes sorted according their detection
                
                simweights = []
                
                for prototype_idx in sorted_pooled_indices:
                    
                    simweight = pooled[0, prototype_idx].item() *\
                        net.module._classification.weight[
                            pred_class_idx, prototype_idx].item()
                    
                    simweights.append(simweight)
                    if abs(simweight) > 0.01:
                        
                        c_weight = torch.max(classification_weights[:, prototype_idx]) 
                        
                        # ignore prototypes that are not relevant to any class
                        if (c_weight > 1e-10):
                            
                            # get the coordinate of the maximum in the feature's space
                            max_hw, max_idx_hw = torch.max(
                                softmaxes[0, prototype_idx, :, :, :], dim=0) # (d,h,w)->(h,w)
                            max_h, max_idx_h = torch.max(max_hw, dim=0) # (h,w)->(w)
                            max_w, max_idx_w = torch.max(max_h, dim=0)  # (w)->(1)

                            w_idx = max_idx_w.item()
                            h_idx = max_idx_h[w_idx].item()
                            d_idx = max_idx_hw[h_idx, w_idx].item()
                            
                            img_np = np.expand_dims(np.load(img),
                                                    axis=0)
                            img_tensor = transforms.RepeatChannel(
                                repeats = args.channels)(img_np)
                            
                            img_tensor = transforms.Resize(
                                spatial_size = (args.slices, 
                                                args.rows, 
                                                args.cols))(img_tensor)
                            
                            # img_tensor: shape (1, 3, slices, rows, cols)
                            img_tensor = img_tensor.unsqueeze_(0) 
                            
                            ps_coord = get_img_coordinates(
                                args.slices, args.rows, args.cols, 
                                softmaxes.shape, 
                                patchsize, skip_z, skip_y, skip_x,
                                d_idx, h_idx, w_idx)
                            
                            local_explanation[
                                prototype_idx.item()] = (ps_coord, simweight)
                            
        local_explanations.append(local_explanation)
        title = "Prediction " + str(ys_pred.item()) + "\n Detected PS: " + str(local_explanation.keys())
        
        if plot:
            plot_local_explanation(xs.cpu(), local_explanation, title=title)
        
    return local_explanations, y_preds, y_trues
                        

@torch.no_grad()
def eval_local_explanations(
        net, 
        local_explanations, 
        device,
        args: argparse.Namespace):
    """
    Compute functionally grounded evaluation metrics """
    
    atlas_path = "/3DPIPNet/data/raw/mni_icbm152_nlin_sym_09c_CerebrA_nifti/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii"
    atlas_image = sitk.ReadImage(atlas_path)
    atlas = sitk.GetArrayFromImage(atlas_image)
    atlas = crop_brain(atlas)
    atlas = np.expand_dims(atlas, axis=0)
    
    atlas_tensor = transforms.RepeatChannel(repeats = args.channels)(atlas)
    atlas_tensor = transforms.Resize(
        spatial_size = (args.slices, 
                        args.rows, 
                        args.cols),
        mode="nearest-exact")(atlas_tensor)
    atlas_tensor = atlas_tensor.unsqueeze_(0) # (1,3,slices,rows,cols)
    atlas = np.array(atlas_tensor)
    
    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    relevant_ps = [ps.item() for ps in classification_weights.nonzero(as_tuple=True)[1]] 
    
    ps_cc_coords = {ps:[] for ps in relevant_ps}
    ps_scores = {ps:[] for ps in relevant_ps}
    ps_entropy = {ps:[] for ps in relevant_ps}
    
    for i, local_explanation in enumerate(local_explanations):
        
        proto_found = list(local_explanation.keys())
        proto_found.sort()
        proto_not_found = list(set(relevant_ps)-set(local_explanation.keys()))
        proto_not_found.sort()
        
        for ps in proto_found:
            dmin, dmax, hmin, hmax, wmin, wmax = local_explanation[ps][0]
            ps_cc_d = (dmin + dmax)/2
            ps_cc_h = (hmin + hmax)/2
            ps_cc_w = (wmin + wmax)/2
            ps_cc = np.array([ps_cc_d, ps_cc_h, ps_cc_w])
            ps_cc_coords[ps].append(ps_cc)
            ps_scores[ps].append(local_explanation[ps][1])
            ps_atlas = atlas[0, 0, dmin:dmax, hmin:hmax, wmin:wmax]
            ps_dim = len(ps_atlas.flatten())
            brain_regions_included, brain_regions_counts = np.unique(
                ps_atlas, axis=None, return_counts=True)
            ps_entropy[ps].append(entropy(brain_regions_counts/ps_dim, base=2))
        
        for ps in proto_not_found:
            ps_cc_coords[ps].append(None)
            ps_scores[ps].append(None)
            ps_entropy[ps].append(None)
    
    ps_scores_df = pd.DataFrame(ps_scores)
    
    # How many times each prototype is detected dataset
    ps_detections = ps_scores_df.count()
    
    # Variation in prototypes' localization
    mean_ps_coords = dict()
    std_ps_coords = dict()
    mean_ps_entropy = dict() # ps entropy (purity index)
    lc_ps = dict() # ps localizations' consistency
    
    for ps in relevant_ps: 
        
        mean_ps_coords[ps] = np.array([
            ps_coord for ps_coord in ps_cc_coords[ps] if ps_coord is not None]
            ).mean(0) # mean of ps coordinates 
        
        std_ps_coords[ps] = np.array([
            ps_coord for ps_coord in ps_cc_coords[ps] if ps_coord is not None]
            ).std(0) # std of ps' coordinates
        
        mean_ps_entropy[ps] = np.round(np.array([
            ps_coord for ps_coord in ps_entropy[ps] if ps_coord is not None]
            ).mean(0), decimals=2)
        
        lc_ps[ps] = np.array([
            np.linalg.norm(ps_coord-mean_ps_coords[ps])/(np.sqrt(3)*32) for ps_coord in ps_cc_coords[ps] if ps_coord is not None]
            ).mean(0)
    
    return ps_detections, mean_ps_coords, std_ps_coords, mean_ps_entropy, lc_ps#, ps_brain_distribution_cn, ps_brain_distribution_ad


@torch.no_grad()
# Calculates class-specific threshold for the FPR@X metric. 
# Also calculates threshold for images with correct prediction 
# (currently not used, but can be insightful)
def get_thresholds(net,
        test_loader: DataLoader,
        epoch,
        device,
        percentile: float = 95.,
        log: Log = None,  
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
    
    outputs_per_class = dict()
    outputs_per_correct_class = dict()
    for c in range(net.module._num_classes):
        outputs_per_class[c] = []
        outputs_per_correct_class[c] = []
        
    # Show progress on progress bar
    test_iter = iter(test_loader)
    
    # Iterate through the test set
    for i, (xs, ys) in enumerate(test_iter):
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, pooled, out = net(xs)

            ys_pred = torch.argmax(out, dim=1)
            for pred in range(len(ys_pred)):
                outputs_per_class[
                    ys_pred[pred].item()].append(out[pred,:].max().item())
                
                if ys_pred[pred].item()==ys[pred].item():
                    outputs_per_correct_class[
                        ys_pred[pred].item()].append(out[pred,:].max().item())
        
        del out
        del pooled
        del ys_pred

    class_thresholds = dict()
    correct_class_thresholds = dict()
    all_outputs = []
    all_correct_outputs = []
    
    for c in range(net.module._num_classes):
        if len(outputs_per_class[c])>0:
            outputs_c = outputs_per_class[c]
            all_outputs += outputs_c
            class_thresholds[c] = np.percentile(outputs_c, 100-percentile) 
            
        if len(outputs_per_correct_class[c])>0:
            correct_outputs_c = outputs_per_correct_class[c]
            all_correct_outputs += correct_outputs_c
            correct_class_thresholds[c] = np.percentile(correct_outputs_c, 
                                                        100-percentile)
    
    overall_threshold = np.percentile(all_outputs,
                                      100-percentile)
    
    overall_correct_threshold = np.percentile(all_correct_outputs,
                                              100-percentile)
    
    # if class is not predicted there is no threshold. 
    # we set it as the minimum value for any other class 
    mean_ct = np.mean(list(class_thresholds.values()))
    mean_cct = np.mean(list(correct_class_thresholds.values()))
    
    for c in range(net.module._num_classes):
        
        if c not in class_thresholds.keys():
            print(c,"not in class thresholds. Setting to mean threshold", 
                  flush=True)
            class_thresholds[c] = mean_ct
            
        if c not in correct_class_thresholds.keys():
            correct_class_thresholds[c] = mean_cct

    calculated_percentile = 0
    correctly_classified = 0
    total = 0
    
    for c in range(net.module._num_classes):
        correctly_classified+=sum(i>class_thresholds[c] for i in outputs_per_class[c])
        total += len(outputs_per_class[c])
        
    calculated_percentile = correctly_classified/total

    if percentile < 100:
        
        while calculated_percentile < (percentile/100.):
            class_thresholds.update(
                (x, y*0.999) for x, y in class_thresholds.items())
            correctly_classified = 0
            
            for c in range(net.module._num_classes):
                correctly_classified+=sum(i>=class_thresholds[c] for i in outputs_per_class[c])
            calculated_percentile = correctly_classified/total

    return overall_correct_threshold, overall_threshold, correct_class_thresholds, class_thresholds


@torch.no_grad()
def eval_ood(net,
        test_loader: DataLoader,
        epoch,
        device,
        threshold, # class specific threshold or overall threshold. single float is overall, list or dict is class specific 
        progress_prefix: str = 'Get Thresholds Epoch'
        ) -> dict:
    
    net = net.to(device)
    # Make sure the model is in evaluation mode
    net.eval()   
 
    predicted_as_id = 0
    seen = 0.
    abstained = 0

    test_iter = iter(test_loader)
    
    # Iterate through the test set
    for i, (xs, ys) in enumerate(test_iter):
        xs, ys = xs.to(device), ys.to(device)
        
        with torch.no_grad():
            # Use the model to classify this batch of input data
            _, pooled, out = net(xs)
            max_out_score, ys_pred = torch.max(out, dim=1)
            ys_pred = torch.argmax(out, dim=1)
            abstained += (max_out_score.shape[0] - torch.count_nonzero(max_out_score))
            
            for j in range(len(ys_pred)):
                seen+=1.
                if isinstance(threshold, dict):
                    thresholdj = threshold[ys_pred[j].item()]
                elif isinstance(threshold, float): #overall threshold
                    thresholdj = threshold
                else:
                    raise ValueError(
                        "provided threshold should be float or dict", 
                        type(threshold))
                sample_out = out[j,:]
                
                if sample_out.max().item() >= thresholdj:
                    predicted_as_id += 1
                    
            del out
            del pooled
            del ys_pred
            
    print(
        "Samples seen:", seen, "of which predicted as In-Distribution:", 
        predicted_as_id, 
        flush=True)
    print(
        "PIP-Net abstained from a decision for", abstained.item(), "images", 
        flush=True)
    
    return predicted_as_id/seen
    
    
    
    
    
    
    
    