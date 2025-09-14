#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:58:53 2025

@author: linuxu
"""

# # Goal
# The goal is to use a simple model to classify x-ray images in Keras, the notebook how to use the ```flow_from_dataframe``` to deal with messier datasets

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau



# In[2]:


all_xray_df = pd.read_csv('./Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('.', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
#all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(x[:-1]))
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: int(str(x)[:-1]) if isinstance(x, str) else x)
all_xray_df.sample(3)


# # Preprocessing Labels
# Here we take the labels and make them into a more clear format. The primary step is to see the distribution of findings and then to convert them to simple binary labels

# In[3]:


label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[4]:


all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df.sample(3)


# ### Clean categories
# Since we have too many categories, we can prune a few out by taking the ones with only a few examples

# In[5]:


# keep at least 1000 cases
MIN_CASES = 1000
all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
print('Clean Labels ({})'.format(len(all_labels)), 
      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])


# In[6]:

# Calculate the sample weights for each row in the DataFrame
# The weight is based on the number of findings (split by '|') in the 'Finding Labels' column: weight is 0.1 + number of findings

sample_weights = all_xray_df['Finding Labels'].map(
    lambda x: max(0, 1.0 - len(x.split('|')) * 0.1)  # Inverse of number of findings (more findings = lower weight)
).values + 4e-2  # Add a small constant (0.04) to avoid zero weights for rows with no findings

# sample_weights = all_xray_df['Finding Labels'].map(
#     lambda x: len(x.split('|')) if len(x) > 0 else 0  
# ).values + 4e-2  

# Normalize the weights so that they sum to 1
sample_weights /= sample_weights.sum()


# Perform weighted random sampling of 40,000 rows from the DataFrame
# Rows with higher weights (fewer findings) have a higher chance of being selected
all_xray_df = all_xray_df.sample(100000, weights=sample_weights)


label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
_ = ax1.set_xticklabels(label_counts.index, rotation = 90)


# In[7]:


label_counts = 100*np.mean(all_xray_df[all_labels].values,0)
fig, ax1 = plt.subplots(1,1,figsize = (12, 8))
ax1.bar(np.arange(len(label_counts))+0.5, label_counts)
ax1.set_xticks(np.arange(len(label_counts))+0.5)
ax1.set_xticklabels(all_labels, rotation = 90)
ax1.set_title('Adjusted Frequency of Diseases in Patient Group')
_ = ax1.set_ylabel('Frequency (%)')


# # Prepare Training Data
# Here we split the data into training and validation sets and create a single vector (disease_vec) with the 0/1 outputs for the disease status (what the model will try and predict)

# In[8]:


all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
all_xray_df['pathological'] = all_xray_df['disease_vec'].apply(lambda x: 1 if 1.0 in x else 0)
all_xray_df.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/data.csv')

all_xray_df['pathological'].value_counts()

# In[9]:

# train_df, valid_df = train_test_split(all_xray_df, 
#                                    test_size = 0.2, 
#                                    random_state = 42,
#                                    stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4])) #ensure that both the training and validation sets have a similar number of examples from diseases 
# train_df = train_df.reset_index(drop=True)
# valid_df = valid_df.reset_index(drop=True)
# print('train', train_df.shape[0], 'validation', valid_df.shape[0])


train_df, temp_df = train_test_split(all_xray_df, 
                                     test_size=0.2, 
                                     random_state=42,
                                     stratify=all_xray_df['Finding Labels'].map(lambda x: x[:4]))  # Stratify based on 'Finding Labels'

# Second split: split the 20% temporary data into 50% validation and 50% test (10% each)
valid_df, test_df = train_test_split(temp_df, 
                                     test_size=0.5, 
                                     random_state=42,
                                     stratify=temp_df['Finding Labels'].map(lambda x: x[:4]))  # Stratify the split as well

# Reset the indices
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Check the distributions
print("Training set distribution:\n", train_df.shape[0])
print("\nValidation set distribution:\n", valid_df.shape[0])
print("\nTest set distribution:\n", test_df.shape[0])

# train_df.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/train_df.csv')
# valid_df.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/valid_df.csv')
# test_df.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/test_df.csv')


train_subset = train_df.sample(n=100, random_state=42)
test_subset = train_subset.copy()
#test_subset = test_df.sample(n=100, random_state=42)

#%% Create Data Generators

# class xray_dataset(Dataset):
#     def __init__(self, metadata_df, device):
#         self.metadata_df = metadata_df
#         self.device = device

#     def __len__(self):
#         return len(self.metadata_df)

#     def __getitem__(self, idx):
#         label = self.metadata_df.iloc[idx]['pathological']
#         filepath = self.metadata_df.iloc[idx]['path']
#         img = cv2.imread(filepath)
#        # img = cv2.resize(img, (128, 128))  # Resize the image to 128x128
#         img = cv2.resize(img, (224, 224))  # Resize to 224x224 for ResNet50
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img_tensor = torch.tensor(img, dtype=torch.float32)
#         img_tensor = img_tensor.permute(2, 0, 1)  # Change from HxWxC to CxHxW
#         label_tensor = torch.tensor(label, dtype=torch.float32)
#         img_tensor = img_tensor.to(self.device)
#         label_tensor = label_tensor.to(self.device)
#         return img_tensor, label_tensor


import torchvision.transforms as transforms


class xray_dataset(Dataset):
    def __init__(self, metadata_df, device, transform=None):
        self.metadata_df = metadata_df
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        label = self.metadata_df.iloc[idx]['pathological']
        filepath = self.metadata_df.iloc[idx]['path']
        
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Always resize manually

        img = img.astype(np.float32) / 255.0  # Normalize manually [0,1]
        img = torch.tensor(img, dtype=torch.float32)  # Create tensor directly
        img = img.permute(2, 0, 1)  # Change HWC -> CHW (expected input)

        if self.transform:
            img = self.transform(img)  # Now transform safely (pure torch ops)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        img = img.to(self.device)
        label_tensor = label_tensor.to(self.device)
        
        return img, label_tensor


# Transform for Training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Transform for Validation (no random)
valid_transform = transforms.Compose([
    # No need to resize, normalize again
])

class xray_nn(nn.Module):
    def __init__(self, n_output=1):
        super(xray_nn, self).__init__()
        
        # Load a pre-trained ResNet50 model
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer to output 1 value for binary classification
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, n_output)

    def forward(self, x):
        x = self.backbone(x)
        return x
    
#with freeze:
# class xray_nn(nn.Module):
#     def __init__(self, n_output=1, freeze_backbone=True):
#         super(xray_nn, self).__init__()
        
#         # Load a pre-trained ResNet50 model
#         self.backbone = models.resnet50(pretrained=True)
        
#         # Optionally freeze all layers
#         if freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
        
#         # Replace the final fully connected layer
#         num_ftrs = self.backbone.fc.in_features
#         self.backbone.fc = nn.Linear(num_ftrs, n_output)
        
#         # Only the new FC layer will be trainable if freezing

#     def forward(self, x):
#         x = self.backbone(x)
#         return x


def model_train(model, epoch, train_loader, loss_fc, optimizer, ckpt_save_path):
    num_batches = len(train_loader)
    model.train()  # Training phase: Dropout etc. are working
    running_loss = 0.0
    epoch_loss = 0.0
    for i, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        # Apply BCEWithLogitsLoss directly 
        loss = loss_fc(output.squeeze(), target)
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        # Print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        step_str = f'[{epoch}, {i:5d}/{num_batches - 1}]'
        print(f'{step_str} loss: {running_loss:.3f}')
        running_loss = 0.0
       

    # Save checkpoint after the epoch ends
    avg_loss_per_batch = epoch_loss / num_batches
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'avg_loss': avg_loss_per_batch},
        ckpt_save_path / f'train_ckpt_epoch{epoch}.pt')

    return avg_loss_per_batch


def model_test(test_loader, model, loss_fc, epoch, ckpt_save_path, set_type):
    num_batches = len(test_loader)
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():  # No need for gradients in evaluation phase
        for i, (data, target) in enumerate(test_loader):
            output = model(data)  # Output shape: (batch_size, 1) for binary classification
            test_loss += loss_fc(output.squeeze(), target).item()  # Squeeze to remove singleton dimension
    # Calculate average test loss
    avg_loss_per_batch = test_loss / num_batches
    print(f"Avg {set_type} loss: {avg_loss_per_batch:.3f}")
    torch.save({'loss': avg_loss_per_batch},
               ckpt_save_path / f'{set_type}_ckpt_epoch{epoch}.pt')
    
    return avg_loss_per_batch

                     
    
def model_verify(test_loader, model, loss_fc, epoch, ckpt_save_path, set_type):
    res_df = pd.DataFrame()
    num_batches = len(test_loader)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # No need for gradients in evaluation phase
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Convert tensors to plain Python lists (avoiding .numpy())
            output_list = output.squeeze().detach().cpu().tolist()
            target_list = target.detach().cpu().tolist()

            # If batch size is 1, wrap in list to avoid scalar issues
            if isinstance(output_list, float):
                output_list = [output_list]
                target_list = [target_list]

            res_df_i = pd.DataFrame({'target': target_list, 'output': output_list})
            res_df = pd.concat([res_df, res_df_i], ignore_index=True)

    return res_df


#%% main:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epoch = 20
ckpt_save_path = Path(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug_w_sched')


train_dataset = xray_dataset(train_df, device, transform = train_transform)
valid_dataset = xray_dataset(valid_df, device,  transform = valid_transform)
test_dataset = xray_dataset(test_df, device,  transform = valid_transform)


batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = xray_nn()
model.to(device)

loss = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.Adam(params=model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        model_train(model, epoch, train_dataloader, loss, optimizer, ckpt_save_path)
        model_test(test_dataloader, model, loss, epoch, ckpt_save_path, 'validation')
        
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        model_train(model, epoch, train_dataloader, loss, optimizer, ckpt_save_path)
        val_loss = model_test(test_dataloader, model, loss, epoch, ckpt_save_path, 'validation')
        # Step the scheduler with validation loss
        scheduler.step(val_loss)
        pbar.update(1)

        
#%% results for selected models:
   
def model_verify(test_loader, model, loss_fc, epoch, ckpt_save_path, set_type, device):
    """
    Verifies a model by running it on the test set and collecting predictions and targets.
    """
    res_df = pd.DataFrame()
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Ensure outputs and targets are 1D lists
            output_list = output.squeeze().detach().cpu().tolist()
            target_list = target.squeeze().detach().cpu().tolist()

            # Wrap in list if scalar (batch size 1)
            if isinstance(output_list, float):
                output_list = [output_list]
                target_list = [target_list]

            batch_df = pd.DataFrame({'target': target_list, 'output': output_list})
            res_df = pd.concat([res_df, batch_df], ignore_index=True)

    #save results
    res_df.to_csv(ckpt_save_path / f'{set_type}_results_epoch{epoch}.csv', index=False)
    return res_df

# Set path and device
selected_ckpt = Path(r'//home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug/train_ckpt_epoch16.pt')


# Load model and optimizer
model = xray_nn()
checkpoint = torch.load(selected_ckpt, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Define loss function and DataLoader
loss_fc = nn.BCEWithLogitsLoss()

res_df = model_verify(
    test_loader=test_dataloader,
    model=model,
    loss_fc=loss_fc,
    epoch=16,
    ckpt_save_path=Path('/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/all_results'),
    set_type='test',
    device=device)

res_df.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug/res_model16_n.csv')

#%% results for all models in a folder:

ckpt_folder = Path('/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/all_results/models')
output_folder = Path('/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/all_results')
output_folder.mkdir(parents=True, exist_ok=True)

# Define loss function
loss_fc = nn.BCEWithLogitsLoss()

# Iterate over all checkpoint files
for ckpt_file in ckpt_folder.glob('train_ckpt_epoch*.pt'):
    epoch_str = ckpt_file.stem.split('epoch')[-1]
    try:
        epoch = int(epoch_str)
    except ValueError:
        print(f"Skipping invalid checkpoint filename: {ckpt_file}")
        continue

    print(f"Processing model from epoch {epoch}...")

    # Build and load model
    model = xray_nn()
    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Run model_verify
    model_verify(
        test_loader=test_dataloader,
        model=model,
        loss_fc=loss_fc,
        epoch=epoch,
        ckpt_save_path=output_folder,
        set_type='test',
        device=device
    )



#%% results visualization:
import torch
import matplotlib.pyplot as plt
import re
import os
#results = torch.load(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results/validation_ckpt_epoch1.pt')

# Directory where your .pt files are stored
ckpt_dir = '/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug_w_sched'  # <-- change this if needed

# Pattern matchers
train_pattern = re.compile(r'train_ckpt_epoch(\d+)\.pt')
val_pattern = re.compile(r'validation_ckpt_epoch(\d+)\.pt')

# Containers for data
train_losses = {}
val_losses = {}

# Read and parse files
for fname in os.listdir(ckpt_dir):
    fpath = os.path.join(ckpt_dir, fname)
    if train_pattern.match(fname):
        epoch = int(train_pattern.match(fname).group(1))
        loss = torch.load(fpath)['avg_loss']
        train_losses[epoch] = loss
    elif val_pattern.match(fname):
        epoch = int(val_pattern.match(fname).group(1))
        loss = torch.load(fpath)['loss']
        val_losses[epoch] = loss

# Sort by epoch
train_epochs = sorted(train_losses.keys())
val_epochs = sorted(val_losses.keys())

train_loss_values = [train_losses[e] for e in train_epochs]
val_loss_values = [val_losses[e] for e in val_epochs]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_loss_values, label='Training Loss', marker='o')
plt.plot(val_epochs, val_loss_values, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ROC plot:
    
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
res_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug/res_model20.csv')

# Calculate False Positive Rate, True Positive Rate, and thresholds
fpr, tpr, thresholds = roc_curve(res_df['target'], res_df['output'])
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# all results:
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

# Folder containing result CSVs
results_folder = '/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/all_results'

# Storage for evaluation metrics
summary_metrics = []

# Loop over all CSV files
for filename in os.listdir(results_folder):
    if filename.endswith(".csv"):
        model_name = filename.replace(".csv", "")
        file_path = os.path.join(results_folder, filename)
        
        # Read prediction data
        df = pd.read_csv(file_path)
        if 'target' not in df.columns or 'output' not in df.columns:
            print(f"Skipping {filename}: required columns missing.")
            continue

        y_true = df['target']
        y_pred_proba = df['output']
        y_pred = (y_pred_proba >= 0.5).astype(int)  # Threshold at 0.5

        # Compute metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Append to summary
        summary_metrics.append({
            'model': model_name,
            'AUC': roc_auc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        })

        # Plot ROC curve
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', color='darkorange')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()  # or save using plt.savefig()

# Convert summary to DataFrame
summary_df = pd.DataFrame(summary_metrics)

#%%
import torch
from pathlib import Path

# Path to your .pth file
checkpoint_path = Path("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug_w_sched/train_ckpt_epoch20.pt")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Use 'cuda' if needed

# Inspect keys
print("Checkpoint keys:", checkpoint.keys())

# Extract training info
epoch = checkpoint['epoch']
avg_loss = checkpoint['avg_loss']
print(f"Epoch: {epoch}, Avg Loss: {avg_loss:.4f}")

# Model weights
model_weights = checkpoint['model_state_dict']

# Optimizer details (e.g., learning rate, weight decay)
optimizer_state = checkpoint['optimizer_state_dict']
for group in optimizer_state['param_groups']:
    print("Learning rate:", group['lr'])
    print("Weight decay:", group.get('weight_decay', 'N/A'))
