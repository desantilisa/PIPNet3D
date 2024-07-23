import os
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from typing import Dict, List, Tuple


def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device) -> Tuple[float, float]:
    
    """ Trains a PyTorch model for a single epoch.
    
    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").

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
        y_pred_prob = F.softmax(y_logit, dim=-1) # predicted probability
        y_pred_class = torch.argmax(y_pred_prob, dim=-1) # predicted class label
        
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_logit, y)
        train_loss += loss.item() 
    
        # 3. Optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backward
        loss.backward()
    
        # 5. Optimizer step
        optimizer.step()
    
        # Calculate and accumulate accuracy metric across all batches
        train_acc += (y_pred_class == y).sum().item()/len(y)
    
    # Update the learning rate at the end of each epoch
    scheduler.step()
    
    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc


def val_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device) -> Tuple[float, float]:
    
    """ Tests a PyTorch model for a single epoch.
    
    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    
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
            y_pred_prob = F.softmax(y_logit, dim=-1) # predicted probability
            y_pred_class = torch.argmax(y_pred_prob, dim=-1) # predicted class label
            
            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_logit, y)
            val_loss += loss.item()
    
            # Calculate and accumulate accuracy
            val_acc += ((y_pred_class == y).sum().item()/len(y))
    
    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train_blackbox(
        model: torch.nn.Module,
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
    
    """ Trains and tests a PyTorch model.
    
    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      val_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      scheduler: 
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
    
    """
    # Create empty results dictionary
    results = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    
    # Loop through training and testing steps for a number of epochs
    #for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        
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
        print(f"Epoch: {epoch+1} | " f"train_loss: {train_loss:.4f} | " f"train_acc: {train_acc:.4f} | " f"val_loss: {val_loss:.4f} | " f"val_acc: {val_acc:.4f}")
        
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
