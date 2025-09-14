#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:07:55 2025

@author: linuxu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


#%% define classes:
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
    
# NN Class
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
    
# Transform for Training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# Transform for Validation (no random)
valid_transform = transforms.Compose([])
#%% read data:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/train_df.csv')
valid_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/valid_df.csv')
test_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/test_df.csv')


train_dataset = xray_dataset(train_df, device, transform = train_transform)
valid_dataset = xray_dataset(valid_df, device,  transform = valid_transform)
test_dataset = xray_dataset(test_df, device,  transform = valid_transform)


batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#%% AA:
#img = cv2.imread(r"C:\Users\AmitTzahar\OneDrive - Tyto Care Ltd\Documents\afeka\project\00000001_000.png")
from PIL import Image
import torch
from pathlib import Path
all_xray_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/data.csv')
img = Image.open("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/nih/images_001/images/00000001_000.png").convert("RGB")
img_label = all_xray_df[all_xray_df['Image Index']=='00000001_000.png']['pathological'].reset_index(drop=True)
img_label = img_label.to_numpy()[0]
img = img.resize((224, 224))  # Resize

# Convert to torch tensor and normalize
img = torch.tensor(list(img.getdata()), dtype=torch.float32).view(224, 224, 3) / 255.0

# Convert to CHW format
img = img.permute(2, 0, 1)

# For visualization: convert back to HWC
img_to_show = img.permute(1, 2, 0)

# Convert to list of RGB tuples for PIL
pixels = [
    tuple((img_to_show[y, x] * 255).clamp(0, 255).to(torch.uint8).tolist())
    for y in range(224)
    for x in range(224)
]

# Create new image and load data
img_display = Image.new("RGB", (224, 224))
img_display.putdata(pixels)

# Display the image
img_display.show()

# load model:
# Set path and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_ckpt = Path(r'//home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug/train_ckpt_epoch16.pt')

# Load model and optimizer
model = xray_nn()
checkpoint = torch.load(selected_ckpt, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
loss_fc = nn.BCEWithLogitsLoss()    
model.eval()

# attack:
from art.estimators.classification import PyTorchClassifier

classifier = PyTorchClassifier(
    model=model,                             # Your PyTorch model
    loss=nn.BCEWithLogitsLoss()    ,              # Loss function
    input_shape=(3, 224, 224),               # Input shape (C, H, W)
    nb_classes=2,                  # Number of output classes
    clip_values=(0.0, 1.0),                  # Input range
)
    
image = img.unsqueeze(0)  # shape: (1, 3, 224, 224)
#label = torch.tensor([img_label], dtype=torch.long)  
label = torch.tensor([[float(img_label)]], dtype=torch.float32).to(device)
label = label.view(1, 1)  # Reshape to (1, 1)



from art.attacks.evasion import ProjectedGradientDescentPyTorch

attack = ProjectedGradientDescentPyTorch(
    estimator=classifier,
    eps=0.03,             # Max perturbation (L∞)
    eps_step=0.005,       # Step size
    max_iter=40,
    targeted=False,
    num_random_init=0
)

# Convert the PyTorch tensor 'image' to a NumPy array before passing it to the attack
image_numpy = image.detach().cpu().numpy()
label_numpy = label.detach().cpu().numpy()
# Now you can call the attack with the image as a NumPy array
adv_image = attack.generate(x=image_numpy, y=label_numpy)

#new
from art.estimators.classification.pytorch import PyTorchClassifier

# Monkey patch to allow nb_classes=1
PyTorchClassifier._check_params = lambda self: None  # Override internal check
classifier = PyTorchClassifier(
    model=model,
    loss=nn.BCEWithLogitsLoss(),
    input_shape=(3, 224, 224),
    nb_classes=1,  # Now valid because we bypassed the check
    clip_values=(0.0, 1.0)
)
label = torch.tensor([[float(img_label)]], dtype=torch.float32)
image = img.unsqueeze(0)  # shape: (1, 3, 224, 224)

adv_image = attack.generate(x=image.numpy(), y=label.numpy())



#%% another attack
class BinaryToTwoClassWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logit = self.base_model(x)  # shape: [B, 1]
        two_logits = torch.cat([-logit, logit], dim=1)  # shape: [B, 2]
        return two_logits
wrapped_model = BinaryToTwoClassWrapper(model).to(device)
wrapped_model.eval()

def predict_fn(x_numpy: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        logits = wrapped_model(x_tensor)  # shape: [B, 2]
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

from art.estimators.classification import BlackBoxClassifier

classifier = BlackBoxClassifier(
    predict_fn=predict_fn,
    input_shape=(3, 224, 224),
    nb_classes=2,  # ✅ Required, now valid
    clip_values=(0.0, 1.0)
)

from art.attacks.evasion import HopSkipJump

attack = HopSkipJump(
    classifier=classifier,
    max_iter=10,  # Start small
    max_eval=1000,
    init_eval=10,
    targeted=False
)

x_input = img.unsqueeze(0).numpy()
y_input = np.array([img_label])  # 0 or 1

x_adv = attack.generate(x=x_input, y=y_input)

import matplotlib.pyplot as plt
import torch

# Convert to torch tensor and remove batch dimension
adv_tensor = torch.tensor(x_adv[0])  # shape: (3, 224, 224)

# Undo normalization if needed — assuming no ImageNet normalization used
# Just show directly:
plt.imshow(adv_tensor.permute(1, 2, 0).numpy())
plt.title("Adversarial Image")
plt.axis('off')
plt.show()

plt.imshow(img.permute(1, 2, 0).numpy())
plt.title("Original Image")
plt.axis('off')
plt.show()



wrapped_model.eval()
with torch.no_grad():
    adv_input = torch.tensor(x_adv, dtype=torch.float32).to(device)
    logits = wrapped_model(adv_input)         # shape: [1, 2]
    probs = torch.softmax(logits, dim=1)      # shape: [1, 2]
    pred_label = torch.argmax(probs, dim=1)   # shape: [1]

print(f"Predicted label (after attack): {pred_label.item()}")

# Original input prediction
original_input = img.unsqueeze(0).to(device)
with torch.no_grad():
    original_logits = wrapped_model(original_input)
    original_probs = torch.softmax(original_logits, dim=1)
    original_pred = torch.argmax(original_probs, dim=1)

print(f"Original label: {img_label}")
print(f"Predicted before attack: {original_pred.item()}")
print(f"Predicted after attack:  {pred_label.item()}")

#%% multipule:
import os
from glob import glob

def load_image_tensor(img_path: str) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(224, 224, 3) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # CHW
    return img_tensor

results = []

# Adjust to however many samples you want
#sample_df = all_xray_df.sample(n=5, random_state=42).reset_index(drop=True)
sample_df = all_xray_df.copy()
sample_df_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('.', 'images*', '*', '*.png'))}
sample_df['path'] = sample_df['Image Index'].map(sample_df_paths.get)

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

# ---------- Metric Functions ----------
def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))  # assumes input in range [0, 1]

from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    try:
        img1_np = img1.permute(1, 2, 0).cpu().numpy()
        img2_np = img2.permute(1, 2, 0).cpu().numpy()

        # Ensure image is at least 7×7 and valid
        h, w, _ = img1_np.shape
        win_size = min(7, h, w)  # reduce window size if needed
        if win_size % 2 == 0:  # must be odd
            win_size -= 1

        if win_size < 3:
            raise ValueError("Image too small for SSIM")

        return ssim(img1_np, img2_np, channel_axis=-1, data_range=1.0, win_size=win_size)
    except Exception as e:
        print(f"⚠️ SSIM error: {e}")
        return np.nan

# ---------- Metric Accumulators ----------
psnr_scores = []
ssim_scores = []

for idx, row in sample_df.iterrows():
    try:
        img_path = row['path']
        label = int(row['pathological'])

        img_tensor = load_image_tensor(img_path)
        x_input = img_tensor.unsqueeze(0).numpy()
        y_input = np.array([label])

        x_adv = attack.generate(x=x_input, y=y_input)

        adv_tensor = torch.tensor(x_adv[0])

        # Predict on original and adversarial
        with torch.no_grad():
            original_logits = wrapped_model(img_tensor.unsqueeze(0).to(device))
            original_pred = torch.argmax(torch.softmax(original_logits, dim=1), dim=1).item()

            adv_logits = wrapped_model(adv_tensor.unsqueeze(0).to(device))
            adv_pred = torch.argmax(torch.softmax(adv_logits, dim=1), dim=1).item()

        # Compute metrics
        psnr_val = compute_psnr(img_tensor, adv_tensor)
        ssim_val = compute_ssim(img_tensor, adv_tensor)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

        results.append({
            "image": img_path,
            "true_label": label,
            "pred_before_attack": original_pred,
            "pred_after_attack": adv_pred,
            "attack_success": int(original_pred != adv_pred),
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

        print(f"[{idx}] ✔️ {img_path} | PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

    except Exception as e:
        print(f"[{idx}] ⚠️ Failed on {img_path}: {e}")

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

print("\n📊 Overall Perturbation Metrics")
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")

df_results = pd.DataFrame(results)
#%%
df_results.to_csv("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/encryption/attack_results.csv", index=False)
print(df_results)
plt.imshow(adv_tensor.permute(1, 2, 0).numpy())
plt.title("Adversarial Image")
plt.axis('off')
plt.show()


import matplotlib.pyplot as plt

def show_comparison(original_tensor: torch.Tensor, adversarial_tensor: torch.Tensor, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Original image
    axes[0].imshow(original_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Adversarial image
    axes[1].imshow(adversarial_tensor.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Adversarial")
    axes[1].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()
    
    
show_comparison(img_tensor, adv_tensor)
#%% analyze results
import pandas as pd

# Load the results DataFrame
df_results = pd.read_csv("attack_results.csv")  # Update path if needed

# Basic summary
total_samples = len(df_results)
successful_attacks = df_results['attack_success'].sum()
success_rate = successful_attacks / total_samples * 100

# How many samples were originally predicted correctly
originally_correct = (df_results['true_label'] == df_results['pred_before_attack']).sum()

# How many of those were flipped by the attack
correct_flipped_to_wrong = ((df_results['true_label'] == df_results['pred_before_attack']) & 
                            (df_results['true_label'] != df_results['pres_after_attack'])).sum()

# How many predictions were already wrong and stayed wrong
wrong_before_and_after = ((df_results['true_label'] != df_results['pred_before_attack']) & 
                          (df_results['pred_before_attack'] == df_results['pres_after_attack'])).sum()

# How many predictions flipped in total
total_flips = (df_results['pred_before_attack'] != df_results['pres_after_attack']).sum()

# Detailed breakdown
print("🔍 Adversarial Attack Analysis")
print(f"Total Samples: {total_samples}")
print(f"Successful Attacks: {successful_attacks}")
print(f"Attack Success Rate: {success_rate:.2f}%")
print(f"Originally Correct Predictions: {originally_correct}")
print(f"Correct → Incorrect Due to Attack: {correct_flipped_to_wrong}")
print(f"Predictions Flipped (Any Direction): {total_flips}")
print(f"Wrong Before and After (unchanged): {wrong_before_and_after}")

import matplotlib.pyplot as plt

# Pie chart of attack success
labels = ['Successful Attacks', 'Failed Attacks']
sizes = [successful_attacks, total_samples - successful_attacks]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140)
plt.title("Adversarial Attack Success Rate")
plt.axis('equal')
plt.show()
