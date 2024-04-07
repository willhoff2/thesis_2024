#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Trim down imports to only neccesary 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from transformers import AutoFeatureExtractor, ViTForImageClassification, ViTModel
import torch
from torchvision.transforms import v2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('default')

from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler

import datasets_1 as datasets

import os
import re
from sklearn.model_selection import GridSearchCV

import itertools
import torch.optim as optim

from datetime import date
from torch.optim.lr_scheduler import StepLR

from simple_datasets import load_data


# In[10]:


## Pretrained model

class CustomViTEmbeddingModel(torch.nn.Module):
    def __init__(self, original_model):
        super(CustomViTEmbeddingModel, self).__init__()
        
        # Extract the necessary layers from the original model
        self.embeddings = original_model.vit.embeddings  #.patch_embeddings
        self.encoder_layer_0 = original_model.vit.encoder.layer[0]
        self.encoder_layer_1 = original_model.vit.encoder.layer[1]
        
        # Assume a square grid of patches to reshape the sequence of patches back into a 2D grid
            ## image: 224x224 ; patch size: 16x16 --> 14x14 
        self.num_patches_side = 14

    def forward(self, x):
        # Apply the embeddings layer
        x = self.embeddings(x)
        
        # Pass the result through the first and second encoder layers
        x = self.encoder_layer_0(x)[0]  # [0] to get the hidden states
        x = self.encoder_layer_1(x)[0]  # [0] to get the hidden states
        
        # x is now the sequence of embeddings for the patches
            # The output x will be a sequence of embeddings, one for each patch of the input images.
            # If you're looking for a single vector representation per image, typically the class token embedding (the first token) is used. 
            # If the model doesn't use a class token, you might need to apply a different pooling strategy over the patch embeddings.
        
        ## Updating to reshape
        
        # Before reshaping, x is in shape [batch_size, num_patches+1, embedding_dim]
        # We discard the first token which is used for classification in the original ViT model
        x = x[:, 1:, :]  # Now in shape [batch_size, num_patches, embedding_dim]
        
        # Reshape to [batch_size, num_patches_side, num_patches_side, embedding_dim]
        x = x.reshape(-1, self.num_patches_side, self.num_patches_side, x.size(-1))

        # Permute to get [batch_size, embedding_dim, num_patches_side, num_patches_side]
        # This is a pseudo-spatial 2D grid, where embedding_dim becomes the channel dimension
        x = x.permute(0, 3, 1, 2)
        
        return x
    

def calculate_rmse_and_r2(loader, model, scaler, scaled = True):
    ## Should probably make device a param for consistency
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for batch in loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            embeddings = custom_model(images)  # Get embeddings from the ViT
            preds = model(embeddings)  # Pass embeddings to the CNN
            predictions.extend(preds.view(-1).cpu().numpy())  # Transfer to CPU for numpy compatibility
            targets.extend(labels.cpu().numpy())  # Transfer to CPU for numpy compatibility

    # Convert to tensors
    predictions = torch.tensor(predictions).to(device)
    targets = torch.tensor(targets).to(device)

    if scaled:
        # Move targets to CPU for scaling, as scaler works on CPU
        targets_scaled = scaler.transform(targets.cpu().numpy().reshape(-1, 1)).flatten()
        targets_scaled = torch.tensor(targets_scaled).to(device)  # Move back to GPU

        # Calculate RMSE on scaled targets
        rmse_value = torch.sqrt(nn.functional.mse_loss(predictions, targets_scaled))

        # Calculate R^2 on scaled targets (requires CPU operation)
        r2_value = r2_score(targets_scaled.cpu(), predictions.cpu())
    else:
        # Calculate RMSE on unscaled targets
        rmse_value = torch.sqrt(nn.functional.mse_loss(predictions, targets))

        # Calculate R^2 on unscaled targets (requires CPU operation)
        r2_value = r2_score(targets.cpu(), predictions.cpu())

    return rmse_value.item(), r2_value


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device for VIT: {device}")
# Load the pre-trained ViT model
pretrained_vit = ViTForImageClassification.from_pretrained('facebook/deit-tiny-patch16-224').to(device)

## Freeze params
for param in pretrained_vit.parameters():
    param.requires_grad = False
    

# Create model w first three layers and create embedding
custom_model = CustomViTEmbeddingModel(pretrained_vit).to(device)


# In[5]:


## Preventing model arch from printing everytime
InteractiveShell.ast_node_interactivity = "last_expr"
warnings.filterwarnings('ignore')


# In[11]:


## Grid Search
    ## Second grid search

class GridRegressionCNN(nn.Module):
    def __init__(self, embedding_dim, layer_sizes, use_dropout=True, use_batch_norm=True, act_func=nn.ReLU()):
        super(GridRegressionCNN, self).__init__()
        
        layers = []
        in_channels = embedding_dim

        for size in layer_sizes:
            layers.append(nn.Conv2d(in_channels, size, kernel_size=3, padding=1))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(size))
            
            layers.append(act_func)

            if use_dropout:
                ## Changed to 0.1 on 12/19
                layers.append(nn.Dropout(p=0.1))

            in_channels = size

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        ## convert list into sequential
        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(in_features=layer_sizes[-1], out_features=1)

    def forward(self, x):
        x = self.layers(x)

        ## flattening tensor for fully connected
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc(x)
        return x
    


# In[12]:


# Define the hyperparameter grid

# Function to train and evaluate the model
def train_evaluate_model(params,train_loader,val_loader,scaler, full_labels, scaled = False):
    
    ## Maybe have there be a check for whether were using a gpu?
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device in train_evaluate: {device}")
    
    
    model = GridRegressionCNN(embedding_dim=192, layer_sizes=params['layer_sizes'], use_dropout=params['use_dropout'], use_batch_norm=params['use_batch_norm'],act_func = params['activations']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    loss_function = nn.MSELoss()

    # Define LR scheduler
    if params['use_lr_decay']:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust these parameters as needed
    
    # Reset metrics for each training session
    train_losses = []
    train_rmses = []
    val_rmses = []
    
    for epoch in range(params['epochs']):
        model.train()
        train_loss = 0
        #print(f'epoch {epoch} of {epochs}')
        for batch in train_loader:
            optimizer.zero_grad()
            
            images, labels = batch[0].to(device), batch[1].to(device)
            embeddings = custom_model(images)  # Get embeddings from the ViT
            predictions = model(embeddings)  # Pass embeddings to the CNN
            loss = loss_function(predictions.squeeze(), labels) 
            loss.backward()
            optimizer.step()
            if params['use_lr_decay']:
                scheduler.step()  # Update the learning rate
            train_loss += loss.item()

        # Append average loss and RMSE for this epoch
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_rmse, _ = calculate_rmse_and_r2(train_loader, model, scaler, scaled)
        val_rmse, _ = calculate_rmse_and_r2(val_loader, model, scaler, scaled)
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
    
    ## Changed range_target to mean -- 01/07/24
    range_target = torch.mean(full_labels)
    if scaled:
        normal_rmse = scaler.inverse_transform([[val_rmses[-1]]])[0, 0]
        rmse_perc = (normal_rmse / range_target.item()) * 100
    else:
        rmse_perc = (val_rmses[-1] / range_target.item()) * 100
    print(f'PERFORMANCE: \n params: {params}, \n val_rmse: {val_rmses[-1]} \n rmse_perc: {rmse_perc} \n')

    return {'train_rmse': train_rmses[-1], 'val_rmse': val_rmses[-1], 'rmse_perc': rmse_perc}

def manual_grid_search(lake,target,param_grid, train_loader, val_loader,scaler, labels, scaled = False):
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    results = []
    for i, params in enumerate(all_params):
        epochs = params['epochs']
        print(f'run {i+1} of {len(all_params)}')
        performance = train_evaluate_model(params, train_loader, val_loader,scaler, labels, scaled)
        results.append({'params': params, 'performance': performance})

    # Convert results to a DataFrame for easier handling
    results_df = pd.DataFrame([{
        'epochs': r['params']['epochs'],
        'learning_rate': r['params']['learning_rate'],
        'layer_sizes': r['params']['layer_sizes'],
        'use_dropout': r['params']['use_dropout'],
        'use_batch_norm': r['params']['use_batch_norm'],
        'use_lr_decay': r['params']['use_lr_decay'],
        'activation': r['params']['activations'],
        'train_rmse': r['performance']['train_rmse'],
        'val_rmse': r['performance']['val_rmse'],
        'RMSE %': r['performance']['rmse_perc']
    } for r in results])

    # Save the results to a CSV file
    target = target.replace("/", "-")
    title = "grid_results/" + lake + "/" + target + "_" + date.today().strftime("%d%m%Y_%H%M%S") + '_grid_search_results.csv'
    print(f'saved to {title}')
    results_df.to_csv(title, index=False)


# In[13]:


## Warnings off
## Preventing model arch from printing everytime
InteractiveShell.ast_node_interactivity = "last_expr"
warnings.filterwarnings('ignore')

param_grid = {
    'learning_rate': [0.00000001, 0.0000005, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001],
    'layer_sizes': [[256,64], [128,64], [256,128,64], [512,256,64], [512,256,128,64]],
    'use_dropout': [True],
    'use_batch_norm': [True],
    'use_lr_decay': [True,False],
    'activations': [nn.Sigmoid()],
    'epochs': [100,200,500]
}

targets = ['MBT']
lake = 'VIDs'
for target in targets:
    print(f'BEGIN {target}\n')
    if target == "%TOC":
        train_loader, val_loader, scaler,labels_tensor = load_data('%TOC', lake="both", set="full", scaled=False, return_labels = True)
    else:
        train_loader, val_loader, scaler, labels_tensor = load_data('MBT', lake="both", set="full", scaled=False, return_labels = True)
    
        
    ## Do grid search in function
    manual_grid_search(lake, target, param_grid, train_loader, val_loader,scaler, labels_tensor, scaled = False)
    print(f'FINISHED {target}\n')

