import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
import math
from skimage.metrics import structural_similarity as ssim

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SquareAttack

# ---------- Dataset ----------
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

        img = Image.open(filepath).convert("RGB").resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img.to(self.device), torch.tensor(label, dtype=torch.float32).to(self.device)

# ---------- Model ----------
class xray_nn(nn.Module):
    def __init__(self, n_output=1):
        super(xray_nn, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, n_output)

    def forward(self, x):
        return self.backbone(x)

# ---------- ART Wrapper ----------
class ARTBinaryWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logit = self.base_model(x)  # [B, 1]
        return torch.cat([-logit, logit], dim=1)  # [B, 2]

# ---------- Data ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/train_df.csv")
valid_df = pd.read_csv("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/valid_df.csv")
test_df = pd.read_csv("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/test_df.csv")

transform = transforms.Compose([])
test_dataset = xray_dataset(test_df, device, transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------- Load Model ----------
model = xray_nn().to(device)
ckpt_path = Path("/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug/train_ckpt_epoch16.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
model.eval()

# ---------- Wrap for ART ----------
wrapped_model = ARTBinaryWrapper(model).to(device)
wrapped_model.eval()

loss_fn = nn.CrossEntropyLoss()
classifier = PyTorchClassifier(
    model=wrapped_model,
    loss=loss_fn,
    optimizer=torch.optim.Adam(wrapped_model.parameters(), lr=0.001),
    input_shape=(3, 224, 224),
    nb_classes=2,
    clip_values=(0.0, 1.0)
)

# ---------- Square Attack ----------
square_attack = SquareAttack(estimator=classifier, eps=0.05)

# ---------- Load Images ----------
sample_df = test_df.copy()
#sample_df = sample_df.sample(5)
sample_df_paths = {os.path.basename(x): x for x in glob(os.path.join('.', 'images*', '*', '*.png'))}
sample_df['path'] = sample_df['Image Index'].map(sample_df_paths.get)

def load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    return img_tensor

# ---------- Metrics ----------
def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean().item()
    return float('inf') if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))

def compute_ssim(img1, img2):
    try:
        img1_np = img1.permute(1, 2, 0).cpu().numpy()
        img2_np = img2.permute(1, 2, 0).cpu().numpy()
        h, w, _ = img1_np.shape
        win_size = min(7, h, w)
        if win_size % 2 == 0: win_size -= 1
        if win_size < 3: raise ValueError("Image too small for SSIM")
        return ssim(img1_np, img2_np, channel_axis=-1, data_range=1.0, win_size=win_size)
    except Exception as e:
        print(f"SSIM error: {e}")
        return np.nan

# ---------- Attack Loop ----------
results = []
psnr_scores = []
ssim_scores = []

for idx, row in sample_df.iterrows():
    try:
        img_path = row['path']
        label = int(row['pathological'])

        img_tensor = load_image_tensor(img_path)
        x_input = img_tensor.unsqueeze(0).numpy()
        y_input = np.array([int(label)], dtype=np.int64)

        x_adv = square_attack.generate(x=x_input, y=y_input)
        adv_tensor = torch.tensor(x_adv[0])

        with torch.no_grad():
            orig_logit = model(img_tensor.unsqueeze(0).to(device))
            orig_pred = int((torch.sigmoid(orig_logit) > 0.5).item())

            adv_logit = model(adv_tensor.unsqueeze(0).to(device))
            adv_pred = int((torch.sigmoid(adv_logit) > 0.5).item())

        psnr_val = compute_psnr(img_tensor, adv_tensor)
        ssim_val = compute_ssim(img_tensor, adv_tensor)

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

        results.append({
            "image": img_path,
            "true_label": label,
            "pred_before_attack": orig_pred,
            "pred_after_attack": adv_pred,
            "attack_success": int(orig_pred != adv_pred),
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

        print(f"[{idx}] ✔️ {img_path} | PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

    except Exception as e:
        print(f"[{idx}] ⚠️ Failed on {row['Image Index']}: {e}")

# ---------- Summary ----------
df_results = pd.DataFrame(results)
print("\n📊 Summary")
print(df_results.head())

avg_psnr = np.mean([x for x in psnr_scores if np.isfinite(x)])
avg_ssim = np.mean(ssim_scores)
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")

# ---------- Visualization ----------
def show_comparison(original_tensor, adversarial_tensor, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(adversarial_tensor.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Adversarial")
    axes[1].axis('off')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# Show last example
show_comparison(img_tensor, adv_tensor, title="Square Attack Adversarial Example")
#%% analyze results
import pandas as pd

# Basic summary
total_samples = len(df_results)
successful_attacks = df_results['attack_success'].sum()
success_rate = successful_attacks / total_samples * 100

# How many samples were originally predicted correctly
originally_correct = (df_results['true_label'] == df_results['pred_before_attack']).sum()

# How many of those were flipped by the attack
correct_flipped_to_wrong = ((df_results['true_label'] == df_results['pred_before_attack']) & 
                            (df_results['true_label'] != df_results['pred_after_attack'])).sum()

# How many predictions were already wrong and stayed wrong
wrong_before_and_after = ((df_results['true_label'] != df_results['pred_before_attack']) & 
                          (df_results['pred_before_attack'] == df_results['pred_after_attack'])).sum()

# How many predictions flipped in total
total_flips = (df_results['pred_before_attack'] != df_results['pred_after_attack']).sum()

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