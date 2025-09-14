
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
import os
from glob import glob
from skimage.metrics import structural_similarity as ssim
import math
from art.attacks.evasion import HopSkipJump, FastGradientMethod
from art.estimators.classification import BlackBoxClassifier
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


# load model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_ckpt = Path(r'//home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/classification/resnet/results_resnet50_w_aug/train_ckpt_epoch16.pt')

# Load model and optimizer
model = xray_nn()
checkpoint = torch.load(selected_ckpt, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
loss_fc = nn.BCEWithLogitsLoss()    
model.eval()

#%% AA:
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

# ✅ Wrapper to simulate binary classification as 2-class
class ARTBinaryWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        logit = self.base_model(x)  # shape [B, 1]
        return torch.cat([-logit, logit], dim=1)  # shape [B, 2]

# ✅ Use your original model (no modification)
wrapped_model = ARTBinaryWrapper(model).to(device)
wrapped_model.eval()

# ✅ Create ART-compatible classifier
loss_fn = nn.CrossEntropyLoss()

classifier = PyTorchClassifier(
    model=wrapped_model,
    loss=loss_fn,
    optimizer=torch.optim.Adam(wrapped_model.parameters(), lr=0.001),
    input_shape=(3, 224, 224),
    nb_classes=2,  # ART expects at least 2
    clip_values=(0.0, 1.0)
)

# ✅ Define FGSM attack
#fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.05)
fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.015)

sample_df = test_df[:5].copy()
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
save_img_path = '/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/weak_fgsm/fgsm_eps_0.015'

for idx, row in sample_df.iterrows():
    try:
        img_path = row['path']
        label = int(row['pathological'])

        img_tensor = load_image_tensor(img_path)
        x_input = img_tensor.unsqueeze(0).numpy()
        y_input = np.array([int(label)], dtype=np.int64)

        x_adv = fgsm_attack.generate(x=x_input, y=y_input)
        adv_tensor = torch.tensor(x_adv[0])
        
        new_image_path = os.path.join(save_img_path, img_path.lstrip("./").split("/")[-1])

        adv_tensor_to_save = (adv_tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
        adv_tensor_to_save = Image.fromarray(adv_tensor_to_save)
        adv_tensor_to_save.save(new_image_path)

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
show_comparison(img_tensor, adv_tensor, title="FGSM Adversarial Example")

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



