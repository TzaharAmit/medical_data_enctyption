import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm


# dataset class:
class XRayAdvDataset(Dataset):
    """
    Dataset for adversarial training:
      Returns (x_adv, x_clean, y_clean)
        x_adv   : adversarial image (3,224,224) in [0,1]
        x_clean : original clean image (3,224,224) in [0,1]
        y_clean : original clean label (torch.long)
    """
    def __init__(
        self,
        df,
        image_root: str,
        adv_path_col: str = "adv_path",
        clean_path_col: str = "clean_path",
        clean_label_col: str = "label_clean",
        size: int = 224,
    ):
        self.df = df.reset_index(drop=True)
        self.root = image_root
        self.adv_path_col = adv_path_col
        self.clean_path_col = clean_path_col
        self.clean_label_col = clean_label_col
        self.size = size

    def __len__(self):
        return len(self.df)

    def _load_xray_as_tensor(self, filepath):
        """Load grayscale image, resize, scale to [0,1], return (C,H,W) tensor."""
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)   # (H,W)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0               # (H,W) in [0,1]
        img = np.expand_dims(img, axis=-1)                 # (H,W,1)
        img = np.repeat(img, 3, axis=-1)                   # (H,W,3)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C,H,W)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        attacked_path = row[self.adv_path_col]   # path to adversarial image
        original_path = row[self.clean_path_col] # path to original image
        # ensure absolute paths
        attacked_path = attacked_path if os.path.isabs(attacked_path) else os.path.join(self.root, attacked_path)
        original_path = original_path if os.path.isabs(original_path) else os.path.join(self.root, original_path)
        # load tensors
        x_attacked = self._load_xray_as_tensor(attacked_path)
        x_original = self._load_xray_as_tensor(original_path)
        # load original (clean) label
        y_original = torch.tensor(int(row[self.clean_label_col]), dtype=torch.long)
        return x_attacked, x_original, y_original

adv_train_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/df_results.csv')
image_root = r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data'   
adv_train_df['adv_path'] = 'reconstruction/square/train/' + adv_train_df['image'].str.lstrip("./")
adv_train_df['clean_path'] = 'nih/' + adv_train_df['image'].str.lstrip("./") 

adv_test_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/df_test_results.csv')
image_root = r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data'   
adv_test_df['adv_path'] = 'reconstruction/square/train/' + adv_test_df['image'].str.lstrip("./")
adv_test_df['clean_path'] = 'nih/' + adv_test_df['image'].str.lstrip("./") 

adv_train_dataset = XRayAdvDataset(adv_train_df, image_root, 'adv_path', 'clean_path', 'true_label')
adv_test_dataset = XRayAdvDataset(adv_test_df, image_root, 'adv_path', 'clean_path', 'true_label')

batch_size = 8
train_dataloader = DataLoader(adv_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(adv_test_dataset, batch_size=batch_size, shuffle=False)
#%% model:
# ImageNet normalization for the classifier
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[..., None, None]
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[..., None, None]

def imagenet_normalize(x):   # x in [0,1]
    return (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)

def _gaussian_1d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()

def _create_ssim_window(window_size: int, channel: int, device, dtype, sigma: float = 1.5):
    g1d = _gaussian_1d(window_size, sigma, device, dtype).unsqueeze(1)   # (W,1)
    g2d = (g1d @ g1d.t()).unsqueeze(0).unsqueeze(0)                      # (1,1,W,W)
    window = g2d.expand(channel, 1, window_size, window_size).contiguous()  # (C,1,W,W)
    return window

def ssim(img1: torch.Tensor,
         img2: torch.Tensor,
         window_size: int = 11,
         val_range: float = 1.0,
         size_average: bool = True) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) for images in [0,1].
    img1, img2: (B,C,H,W) tensors on same device/dtype.
    Returns mean SSIM over batch if size_average=True, else per-image SSIM (B,).
    """
    assert img1.shape == img2.shape, "SSIM: input shapes must match"
    B, C, H, W = img1.shape
    device, dtype = img1.device, img1.dtype

    window = _create_ssim_window(window_size, C, device, dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        # mean over C,H,W → (B,)
        return ssim_map.mean(dim=(1, 2, 3))

# --- Simple residual denoiser ---
class SimpleDenoiser(nn.Module):
    def __init__(self, ch=64, depth=8, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1)]
            if use_bn: layers += [nn.BatchNorm2d(ch)]
            layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x_adv):
        noise = self.net(x_adv)
        return (x_adv - noise).clamp(0.0, 1.0)

# --- Robust classifier (single-logit) ---
class AdvLabelReconstructor(nn.Module):
    def __init__(self, unfreeze_from="layer4", pretrained=True, dropout=0.1):
        super().__init__()
        m = models.resnet50(pretrained=pretrained)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feats, 1))
        self.backbone = m
        # freeze all, then unfreeze from chosen block + head
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone.fc.parameters(): p.requires_grad = True
        if unfreeze_from is not None:
            turn_on = False
            for name, p in self.backbone.named_parameters():
                if name.startswith(unfreeze_from): turn_on = True
                if turn_on: p.requires_grad = True
    def forward(self, x):   # x should be ImageNet-normalized
        return self.backbone(x)  # (B,1) logits

class JointDenoiseAndClassify(nn.Module):
    """
    One model that:
      - Denoises: x_adv -> x_recon (in [0,1])
      - Classifies: logit = clf( normalize(x_recon) )
    """
    def __init__(self,
                 denoise_ch=64, denoise_depth=8, use_bn=True,
                 cls_unfreeze_from="layer4", cls_pretrained=True, cls_dropout=0.1,
                 freeze_classifier=True):
        super().__init__()
        self.denoiser = SimpleDenoiser(ch=denoise_ch, depth=denoise_depth, use_bn=use_bn)
        self.classifier = AdvLabelReconstructor(unfreeze_from=cls_unfreeze_from,
                                                pretrained=cls_pretrained,
                                                dropout=cls_dropout)
        if freeze_classifier:
            for p in self.classifier.parameters():
                p.requires_grad = False

    def forward(self, x_adv):
        x_recon = self.denoiser(x_adv)                 # [0,1]
        logit   = self.classifier(imagenet_normalize(x_recon))  # (B,1)
        return x_recon, logit

def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction="none").mean(dim=(1,2,3)).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean()

def joint_train_epoch(model, loader, optimizer, device,
                      alpha_l1=0.5, lam_ce=1.0, pos_weight=None):
    model.train()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    tot_loss = tot_acc = 0.0
    n_batches = 0

    for x_adv, x_clean, y in loader:
        x_adv   = x_adv.to(device)
        x_clean = x_clean.to(device)
        y       = y.to(device).float().unsqueeze(1)

        x_recon, logit = model(x_adv)
        # image loss
        l1   = F.l1_loss(x_recon, x_clean)
        ssi  = ssim(x_recon, x_clean)
        img_loss = alpha_l1 * l1 + (1.0 - alpha_l1) * (1.0 - ssi)
        # label loss
        ce = criterion(logit, y)
        loss = img_loss + lam_ce * ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logit) >= 0.5).float()
            acc = (preds == y).float().mean().item()
            tot_loss += loss.item()
            tot_acc  += acc
            n_batches += 1

    return {"loss": tot_loss / n_batches, "acc": tot_acc / n_batches}

@torch.no_grad()
def joint_eval_epoch(model, loader, device,
                     alpha_l1=0.5, lam_ce=1.0, pos_weight=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    tot_loss = tot_acc = 0.0
    n_batches = 0

    for x_adv, x_clean, y in loader:
        x_adv   = x_adv.to(device)
        x_clean = x_clean.to(device)
        y       = y.to(device).float().unsqueeze(1)

        x_recon, logit = model(x_adv)
        l1   = F.l1_loss(x_recon, x_clean)
        ssi  = ssim(x_recon, x_clean)
        img_loss = alpha_l1 * l1 + (1.0 - alpha_l1) * (1.0 - ssi)
        ce = criterion(logit, y)
        loss = img_loss + lam_ce * ce

        preds = (torch.sigmoid(logit) >= 0.5).float()
        acc = (preds == y).float().mean().item()

        tot_loss += loss.item()
        tot_acc  += acc
        n_batches += 1

    return {"loss": tot_loss / n_batches, "acc": tot_acc / n_batches}

#%% train:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = JointDenoiseAndClassify(
    denoise_ch=64, denoise_depth=8, use_bn=True,
    cls_unfreeze_from="layer4", cls_pretrained=True, cls_dropout=0.1,
    freeze_classifier=True  # start frozen; you can unfreeze later if needed
).to(device)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver8_simple_image_label_recon/"
os.makedirs(ckpt_dir, exist_ok=True)

EPOCHS = 10
with tqdm(total=EPOCHS) as pbar:
    for epoch in range(1, EPOCHS + 1):
        tr = joint_train_epoch(model, train_dataloader, optimizer, device,
                               alpha_l1=0.5, lam_ce=1.0, pos_weight=None)
        va = joint_eval_epoch(model, test_dataloader, device,
                              alpha_l1=0.5, lam_ce=1.0, pos_weight=None)

        # --- save metrics (exact keys/filenames) ---
        torch.save({"epoch": int(epoch), "split": "train", "loss": float(tr["loss"]), "acc": float(tr["acc"])},
                   f"{ckpt_dir}/train_metrics_epoch{epoch:03}.pt")
        torch.save({"epoch": int(epoch), "split": "val", "loss": float(va["loss"]), "acc": float(va["acc"])},
                   f"{ckpt_dir}/val_metrics_epoch{epoch:03}.pt")

        # --- save weights (exact filenames) ---
        torch.save(model.state_dict(), f"{ckpt_dir}/train_epoch{epoch:03d}.pt")
        torch.save(model.state_dict(), f"{ckpt_dir}/val_epoch{epoch:03d}.pt")

        print(f"[Joint {epoch}] train loss={tr['loss']:.4f} acc={tr['acc']*100:.1f}% | "
              f"val loss={va['loss']:.4f} acc={va['acc']*100:.1f}%")
        pbar.update(1)

#%% results visualization:
import re, glob
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver8_simple_image_label_recon/"

# ------- helpers -------
epoch_num = lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1))

def files(pattern):
    fs = glob.glob(os.path.join(ckpt_dir, pattern))
    return [p for _, p in sorted((epoch_num(p), p) for p in fs)]

def load_metrics(path):
    # CPU deserialization avoids GPU overhead; only grab the 'loss' field
    obj = torch.load(path, map_location="cpu")
    return float(obj["loss"]), float(obj["acc"])

def load_all_metrics(paths, max_workers=8):
    losses = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(load_metrics, p): i for i, p in enumerate(paths)}
        for f in as_completed(fut):
            losses[fut[f]] = f.result()
    return losses

# ------- gather & load -------
train_paths = files("train_metrics_epoch*.pt")
val_paths   = files("val_metrics_epoch*.pt")

train_metrics = load_all_metrics(train_paths)   # parallel + CPU
val_metrics   = load_all_metrics(val_paths)

train_losses, train_acc= zip(*train_metrics)
val_losses, val_acc= zip(*val_metrics)


# ------- plot -------
plt.figure(figsize=(8,4))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Training", marker="o")
plt.plot(range(1, len(val_losses)+1),   val_losses,   label="Validation", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------- plot -------
plt.figure(figsize=(8,4))
plt.plot(range(1, len(train_acc)+1), train_acc, label="Training", marker="o")
plt.plot(range(1, len(val_acc)+1),   val_acc,  label="Validation", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%% images visualization:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load joint checkpoint ====
ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver8_simple_image_label_recon/train_epoch010.pt"

model = JointDenoiseAndClassify(
    denoise_ch=64, denoise_depth=8, use_bn=True,
    cls_unfreeze_from="layer4", cls_pretrained=True, cls_dropout=0.1,
    freeze_classifier=True  # set to how you trained it
).to(device)

state = torch.load(ckpt_dir, map_location=device)
model.load_state_dict(state)
model.eval()

subset = adv_test_df.head(6)  # change N as you like

for _, row in subset.iterrows():
    # tensors in [0,1] for display
    x_adv = adv_test_dataset._load_xray_as_tensor(os.path.join(image_root, row["adv_path"])).unsqueeze(0).to(device)
    x_cln = adv_test_dataset._load_xray_as_tensor(os.path.join(image_root, row["clean_path"])).unsqueeze(0).to(device)
    y_true     = int(row["true_label"])
    y_attacked = int(row.get("pred_after_attack", -1))

    with torch.no_grad():
        # Joint forward: denoised image + label logit
        x_den, logit = model(x_adv)                   # x_den in [0,1], logit shape (1,1)
        p1      = torch.sigmoid(logit).item()         # P(y=1 | denoised)
        y_recon = int(p1 >= 0.5)
        ssi     = ssim(x_den, x_cln).item()          # quality vs clean

    title_adv = f"ADV (atk={y_attacked if y_attacked!=-1 else 'NA'})"
    title_den = f"DENOISED (recon={y_recon}, SSIM={ssi:.3f})"

    # plot CLEAN | ADV | DENOISED
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    panels = [
        (x_cln[0],  f"CLEAN (true={y_true})"),
        (x_adv[0],  title_adv),
        (x_den[0],  title_den),
    ]
    for ax, (im, ttl) in zip(axes, panels):
        ax.imshow(im.permute(1,2,0).cpu().clamp(0,1))
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
    plt.tight_layout(); plt.show()

#%% joint test-set results (image + label)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

@torch.no_grad()
def model_verify_joint(model, test_loader, adv_test_df, ckpt_dir, device):
    """
    Evaluate the joint model (denoise + classify) on the test set.
    """
    model.eval(); model.to(device)

    # per-image PSNR (expects inputs in [0,1])
    def psnr_per_image(x, y, eps=1e-8):
        mse = torch.mean((x - y) ** 2, dim=(1,2,3)).clamp_min(eps)  # (B,)
        return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()

    has_atk_col = "pred_after_attack" in adv_test_df.columns
    records, idx = [], 0

    for x_adv, x_clean, y in test_loader:
        bsz = x_adv.size(0)
        batch_df = adv_test_df.iloc[idx: idx + bsz]; idx += bsz

        x_adv   = x_adv.to(device)
        x_clean = x_clean.to(device)

        # Forward joint model
        x_den, logit = model(x_adv)                                   # x_den in [0,1], logit (B,1)

        # Label metrics
        probs = torch.sigmoid(logit).squeeze(1).cpu().numpy()          # (B,)
        preds = (probs >= 0.5).astype(int)                             # (B,)

        # Image metrics
        ssim_vals = ssim(x_den, x_clean, size_average=False).cpu().numpy()  # (B,)
        psnr_vals = psnr_per_image(x_den, x_clean)                          # (B,)

        # Collect
        for i in range(bsz):
            row = {
                "adv_path":            batch_df.iloc[i]["adv_path"],
                "clean_path":          batch_df.iloc[i]["clean_path"],
                "true_label":          int(batch_df.iloc[i]["true_label"]),
                "pred_after_recon":    int(preds[i]),
                "prob1_after_recon":   float(probs[i]),
                "ssim":                float(ssim_vals[i]),
                "psnr":                float(psnr_vals[i]),
            }
            if has_atk_col:
                row["pred_after_attack"] = int(batch_df.iloc[i]["pred_after_attack"])
            records.append(row)

    # Save CSV
    df_results = pd.DataFrame(records)
    out_csv = os.path.join(ckpt_dir, "joint_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved {len(df_results)} rows → {out_csv}")

    # Summaries
    y_true = df_results["true_label"].values
    y_pred = df_results["pred_after_recon"].values
    y_prob = df_results["prob1_after_recon"].values

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    mean_ssim   = float(df_results["ssim"].mean())
    median_ssim = float(df_results["ssim"].median())
    mean_psnr   = float(df_results["psnr"].mean())
    median_psnr = float(df_results["psnr"].median())

    summary = {
        "precision": precision, "recall": recall, "f1": f1, "auroc": auroc,
        "mean_ssim": mean_ssim, "median_ssim": median_ssim,
        "mean_psnr": mean_psnr, "median_psnr": median_psnr
    }
    torch.save(summary, os.path.join(ckpt_dir, "joint_summary.pt"))
    print("Summary:", summary)

    # Optional detailed printouts
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    return df_results, summary

#%% run joint eval
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

joint_ckpt = Path(r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver8_simple_image_label_recon/train_epoch010.pt")  

model = JointDenoiseAndClassify(
    denoise_ch=64, denoise_depth=8, use_bn=True,
    cls_unfreeze_from="layer4", cls_pretrained=True, cls_dropout=0.1,
    freeze_classifier=True   # set to how you trained
).to(device)

state = torch.load(joint_ckpt, map_location=device)
model.load_state_dict(state)

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/ver8_simple_image_label_recon/train_epoch010/"
df_results, summary = model_verify_joint(model, test_dataloader, adv_test_df, ckpt_dir, device)
df_results.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/ver8_simple_image_label_recon/train_epoch010/reconstruction_label_results.csv')

#%% all results:
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

df_results = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/ver8_simple_image_label_recon/train_epoch010/reconstruction_label_results.csv')
df_results['image'] = '.' + df_results['clean_path'].str[3:]
adv_test_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/weak_fgsm/train_eps_0.02/df_test_results.csv')
df_results = pd.merge(df_results, adv_test_df[['image', 'pred_before_attack', 'pred_after_attack']])
df_results = df_results[df_results['pred_after_attack']!= df_results['pred_before_attack']]


# Extract true labels and predictions
y_true = df_results['true_label']
y_pred = df_results['pred_after_recon']

# Core metrics
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)


print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

# Full report
print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# (Optional) Visualize confusion matrix
import seaborn as sns
plt.figure(figsize=(3,3))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False,
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

df_results['ssim'].mean()
df_results['psnr'].mean()




