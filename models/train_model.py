import os
import torch
#from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               device: torch.device) -> Tuple[float, float]:
    
    """Trains a PyTorch model for a single epoch.
    
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).
    
    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    
    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:
    
      (0.1112, 0.8743)
    """
    
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
    
        # 1. Forward pass
        y_logit = model(X)
        
        if y_logit.shape[1] == 1:
            y_logit = y_logit.squeeze(dim=1) # logit
            y_pred_prob = torch.sigmoid(y_logit) # predicted probability
            y_pred_class = torch.round(y_pred_prob) # label
            y = y.float()
            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_logit, y)
        else:
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=3)
            y_one_hot = y_one_hot.float()
            y_pred_prob = torch.softmax(y_logit, dim=1) # predicted probability
            y_pred_class = torch.argmax(y_pred_prob, dim=1)
            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_logit, y_one_hot)
            
        train_loss += loss.item() 
    
        # 3. Optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backward
        loss.backward()
    
        # 5. Optimizer step
        optimizer.step()
    
        # Calculate and accumulate accuracy metric across all batches
        train_acc += (y_pred_class == y).sum().item()/len(y_logit)
    
    # Update the learning rate at the end of each epoch
    scheduler.step()
    
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    
    """Tests a PyTorch model for a single epoch.
    
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.
    
    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    
    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (val_loss, val_accuracy). For example:
    
      (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            y_logit = model(X)
            
            if y_logit.shape[1] == 1:
                y_logit = y_logit.squeeze(dim=1) # logit
                y_pred_prob = torch.sigmoid(y_logit) # predicted probability
                y_pred_class = torch.round(y_pred_prob) # label
                y = y.float()
                # 2. Calculate and accumulate loss
                loss = loss_fn(y_logit, y)
            else:
                y_one_hot = torch.nn.functional.one_hot(y, num_classes=3)
                y_one_hot = y_one_hot.float()
                y_pred_prob = torch.softmax(y_logit, dim=1) # predicted probability
                y_pred_class = torch.argmax(y_pred_prob, dim=1)
                # 2. Calculate and accumulate loss
                loss = loss_fn(y_logit, y_one_hot)
                
            val_loss += loss.item()
    
            # Calculate and accumulate accuracy
            val_acc += ((y_pred_class == y).sum().item()/len(y_pred_class))
    
    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train_blackbox(model: torch.nn.Module,
                   train_dataloader: torch.utils.data.DataLoader,
                   val_dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler,
                   loss_fn: torch.nn.Module,
                   epochs: int,
                   device: torch.device,
                   results_path,
                   min_delta = 0.001,
                   patience = 3,
                   current_patience = 0,
                   best_loss = float('inf')) -> Dict[str, List]:
    
    """Trains and tests a PyTorch model.
    
    Passes a target PyTorch models through train_step() and val_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.
    Calculates, prints and stores evaluation metrics throughout.
    Saves the model which achieved the best performances on the validation set.
    
    
    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      val_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      scheduler: 
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    
    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    val_loss: [...],
                    val_acc: [...]} 
      For example if training for epochs=2: 
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    val_loss: [1.2641, 1.5706],
                    val_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }
    
    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
    # for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           scheduler=scheduler,
                                           device=device)
        
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)
    
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )
        
        # Check for improvement
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            current_patience = 0

            # Save the best model's parameters
            model_file = os.path.join(results_path, 'best_model.pth')
            torch.save(model.state_dict(), model_file)
        else:
            current_patience += 1

        # Check if early stopping criteria are met
        # if current_patience >= patience:
        #     print("Early stopping! Training stopped.")
        #     break
    
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
      
    # Return the filled results at the end of the epochs
    return results


def train_pipnet(
        net, 
        train_loader, 
        optimizer_net, 
        optimizer_classifier, 
        scheduler_net, 
        scheduler_classifier, 
        criterion, 
        epoch, 
        nr_epochs, 
        device, 
        pretrain = False, 
        finetune = False, 
        progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    iters = len(train_loader)
    # Show progress on progress bar. 
    # train_iter = tqdm(
    #     enumerate(train_loader),
    #     total=len(train_loader),
    #     desc=progress_prefix+'%s'%epoch,
    #     mininterval=2.,
    #     ncols=0)
    train_iter = enumerate(train_loader)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1  
            
    print("Number of parameters that require gradient: ", 
          count_param, 
          flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 # ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = 5. 
        t_weight = 2.
        unif_weight = 0.
        cl_weight = 2.

    print("Align weight: ", 
          align_pf_weight, 
          ", U_tanh weight: ", 
          t_weight, 
          "Class weight:", 
          cl_weight, 
          flush = True)
    
    print("Pretrain?", pretrain, "Finetune?", finetune, flush = True)
    
    lrs_net = []
    lrs_class = []
    
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs1, xs2, ys) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        
        loss, acc = calculate_loss(
            proto_features, 
            pooled, 
            out, 
            ys, 
            align_pf_weight, 
            t_weight, 
            unif_weight, 
            cl_weight, 
            net.module._classification.normalization_multiplier, 
            pretrain, 
            finetune, 
            criterion, 
            train_iter, 
            print = True, 
            EPS = 1e-8)
        
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()   
            scheduler_classifier.step(epoch - 1 + (i/iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])
     
        if not finetune:
            optimizer_net.step()
            scheduler_net.step() 
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=loss.item()

        if not pretrain:
            
            with torch.no_grad():
                
                net.module._classification.weight.copy_(
                    torch.clamp(
                        net.module._classification.weight.data - 1e-3,
                        min=0.)) # weights < 1e-3 = 0
                
                net.module._classification.normalization_multiplier.copy_(
                    torch.clamp(
                        net.module._classification.normalization_multiplier.data, 
                        min=1.0)) 
                
                if net.module._classification.bias is not None:
                    
                    net.module._classification.bias.copy_(
                        torch.clamp(
                            net.module._classification.bias.data,
                            min=0.))  
                    
    train_info['train_accuracy'] = total_acc/float(i+1)
    train_info['loss'] = total_loss/float(i+1)
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class
    
    return train_info


def calculate_loss(
        proto_features, 
        pooled, 
        out, 
        ys1, 
        align_pf_weight, 
        t_weight, 
        unif_weight, 
        cl_weight, 
        net_normalization_multiplier, 
        pretrain, 
        finetune, 
        criterion, 
        train_iter, 
        print = True, 
        EPS = 1e-10):
    
    ys = torch.cat([ys1, ys1])
    pooled1, pooled2 = pooled.chunk(2) # each one: (bs, num_features)
    pf1, pf2 = proto_features.chunk(2) # each one: (bs, num_features, d, h, w)
    
    # Dimensionality check
    # pf2.flatten(start_dim=2).shape 
    #       -> [bs, num_features, d*h*w]
    # pf2.flatten(start_dim=2).permute(0,2,1).shape 
    #       -> [bs, d*h*w, num_features]
    # pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1).shape 
    #       -> [bs*d*h*w, num_features]
    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (
        align_loss(embv1, embv2.detach()) + 
        align_loss(embv2, embv1.detach()))/2.
    
    tanh_loss = -(
        torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean() + 
        torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean())/2.

    if not finetune:
        loss = align_pf_weight*a_loss_pf
        loss += t_weight * tanh_loss
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs), dim=1), ys)
        
        if finetune:
            loss= cl_weight * class_loss
        else:
            loss+= cl_weight * class_loss
            
    # Our tanh-loss optimizes for uniformity and was sufficient for our 
    # experiments. However, if pretraining of the prototypes is not working 
    # well for your dataset, you may try to add another uniformity loss from 
    # https://www.tongzhouwang.info/hypersphere/ 
    # Just uncomment the following lines
    # else:
    #     uni_loss = (
    #         uniform_loss(F.normalize(pooled1+EPS,dim=1)) + 
    #         uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc = 0.
    
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
    
    return loss, acc


# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. 
# Currently not used but you could try adding it if you want. 
def uniform_loss(x, t=2):
    
    print(
        "sum elements: ", 
        torch.sum(torch.pow(x,2), dim=1).shape, 
        torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss


# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss


