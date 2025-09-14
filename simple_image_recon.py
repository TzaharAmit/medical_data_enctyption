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

batch_size = 16
train_dataloader = DataLoader(adv_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(adv_test_dataset, batch_size=batch_size, shuffle=False)
#%%

class SimpleDenoiser(nn.Module):
    """
    DnCNN-like: Conv-BN-ReLU stacks predict noise n(x_adv); recon = x_adv - n(x_adv).
    Input/Output: (B,3,224,224) in [0,1].
    """
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
        recon = (x_adv - noise).clamp(0.0, 1.0)
        return recon

def psnr(x, y, eps=1e-8):
    # x,y in [0,1], (B,3,H,W)
    mse = F.mse_loss(x, y, reduction="none").mean(dim=(1,2,3)).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean()

import torch
import torch.nn.functional as F

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


def denoiser_train_epoch(model, loader, optimizer, device, alpha_l1=0.5):
    """
    Train one epoch with loss = alpha*L1 + (1-alpha)*(1-SSIM).
    """
    model.train()
    tot_loss = tot_ssim = tot_psnr = 0.0
    n_batches = 0

    for x_adv, x_clean, _ in loader:
        x_adv   = x_adv.to(device)
        x_clean = x_clean.to(device)

        recon = model(x_adv)
        loss_l1   = F.l1_loss(recon, x_clean)
        loss_ssim = 1.0 - ssim(recon, x_clean)
        loss = alpha_l1 * loss_l1 + (1.0 - alpha_l1) * loss_ssim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            tot_loss += loss.item()
            tot_ssim += ssim(recon, x_clean).item()
            tot_psnr += psnr(recon, x_clean).item()
            n_batches += 1

    return {
        "loss": tot_loss / n_batches,
        "ssim": tot_ssim / n_batches,
        "psnr": tot_psnr / n_batches
    }


@torch.no_grad()
def denoiser_eval_epoch(model, loader, device):
    model.eval()
    tot_ssim = tot_psnr = 0.0
    n_batches = 0
    for x_adv, x_clean, _ in loader:
        x_adv   = x_adv.to(device)
        x_clean = x_clean.to(device)
        recon = model(x_adv)
        tot_ssim += ssim(recon, x_clean).item()
        tot_psnr += psnr(recon, x_clean).item()
        n_batches += 1

    return {"ssim": tot_ssim / n_batches, "psnr": tot_psnr / n_batches}
#%% run model training:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

denoiser = SimpleDenoiser(ch=64, depth=8, use_bn=True).to(device)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver7_simple_image_recon/"

EPOCHS = 20  
with tqdm(total=EPOCHS) as pbar:
    best_ssim = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr = denoiser_train_epoch(denoiser, train_dataloader, optimizer, device, alpha_l1=0.5)
        va = denoiser_eval_epoch(denoiser, test_dataloader, device) 

        train_loss = float(tr.get("loss", np.nan))
        val_loss   = float(va.get("loss", np.nan))
        train_acc  = float(tr.get("acc", tr.get("ssim", np.nan)))
        val_acc    = float(va.get("acc", va.get("ssim", np.nan)))

        # --- save metrics (exact filenames/keys) ---
        torch.save(
            {"epoch": int(epoch), "split": "train", "loss": train_loss, "acc": train_acc},
            f"{ckpt_dir}/train_metrics_epoch{epoch:03}.pt"
        )
        torch.save(
            {"epoch": int(epoch), "split": "val", "loss": val_loss, "acc": val_acc},
            f"{ckpt_dir}/val_metrics_epoch{epoch:03}.pt"
        )

        # --- save weights (exact filenames) ---
        torch.save(denoiser.state_dict(), f"{ckpt_dir}/train_epoch{epoch:03d}.pt")
        torch.save(denoiser.state_dict(), f"{ckpt_dir}/val_epoch{epoch:03d}.pt")

        print(f"[Denoiser {epoch}] train loss={train_loss:.4f} acc(SSIM)={train_acc:.3f} | "
              f"val loss={val_loss:.4f} acc(SSIM)={val_acc:.3f}")
        pbar.update(1)
        

#%% results visualization:
import re, glob
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver7_simple_image_recon/"

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
# ==== Load denoiser checkpoint ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

den_ckpt = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver7_simple_image_recon/train_epoch004.pt"   # <-- change to your file
denoiser = SimpleDenoiser(ch=64, depth=8, use_bn=True).to(device)
denoiser.load_state_dict(torch.load(den_ckpt, map_location=device))
denoiser.eval()

subset = adv_test_df.head(6)  # change N as you like

for _, row in subset.iterrows():
    # tensors in [0,1] for display
    x_adv = adv_test_dataset._load_xray_as_tensor(os.path.join(image_root, row["adv_path"])).unsqueeze(0).to(device)
    x_cln = adv_test_dataset._load_xray_as_tensor(os.path.join(image_root, row["clean_path"])).unsqueeze(0).to(device)
    y_true     = int(row["true_label"])
    y_attacked = int(row.get("pred_after_attack", -1))

    with torch.no_grad():
        x_den = denoiser(x_adv)                    # (1,3,224,224)
        ssi   = ssim(x_den, x_cln).item()          # quality vs clean
        # optional label recovery via robust classifier
        title_adv = f"ADV (atk={y_attacked if y_attacked!=-1 else 'NA'})"
        title_den = f"DENOISED (SSIM={ssi:.3f})"

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

#%% test set results:

@torch.no_grad()
def model_verify(denoiser, test_loader, adv_test_df, ckpt_dir, device):
    """
    Evaluate denoiser on the test set (no classifier).
    """
    denoiser.eval(); denoiser.to(device)
    # per-image PSNR helper (expects inputs in [0,1])
    def psnr_per_image(x, y, eps=1e-8):
        # x,y: (B,3,H,W)
        mse = torch.mean((x - y) ** 2, dim=(1,2,3)).clamp_min(eps)  # (B,)
        return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()        # (B,)

    records, idx = [], 0
    for x_adv, x_clean, _ in test_loader:
        bsz = x_adv.size(0)
        batch_df = adv_test_df.iloc[idx: idx + bsz]
        idx += bsz

        x_adv   = x_adv.to(device)
        x_clean = x_clean.to(device)
        x_den = denoiser(x_adv)  # (B,3,H,W) in [0,1]

        # SSIM per-image
        ssim_vals = ssim(x_den, x_clean, size_average=False).cpu().numpy()  # (B,)
        # PSNR per-image
        psnr_vals = psnr_per_image(x_den, x_clean)                          # (B,)

        for i in range(bsz):
            records.append({
                "adv_path":    batch_df.iloc[i]["adv_path"],
                "clean_path":  batch_df.iloc[i]["clean_path"],
                "ssim":        float(ssim_vals[i]),
                "psnr":        float(psnr_vals[i]),
            })

    # Save CSV
    df_results = pd.DataFrame(records)
    out_csv = os.path.join(ckpt_dir, "denoiser_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved {len(df_results)} rows → {out_csv}")

    # Summary
    summary = {
        "mean_ssim": float(df_results["ssim"].mean()),
        "mean_psnr": float(df_results["psnr"].mean()),
        "median_ssim": float(df_results["ssim"].median()),
        "median_psnr": float(df_results["psnr"].median()),
    }
    torch.save(summary, os.path.join(ckpt_dir, "denoiser_summary.pt"))
    print("Summary:", summary)
    return df_results, summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

den_ckpt = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver7_simple_image_recon/train_epoch004.pt"   # <-- change to your file
denoiser = SimpleDenoiser(ch=64, depth=8, use_bn=True).to(device)
denoiser.load_state_dict(torch.load(den_ckpt, map_location=device))

ckpt_dir="/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/all_results/ver7_simple_image_recon/train_epoch004"
df_results, summary = model_verify(denoiser, test_dataloader, adv_test_df, ckpt_dir, device)
df_results.to_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/all_results/ver7_simple_image_recon/train_epoch004/reconstruction_label_results.csv')




#%% get test results: label recon:
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

df_results = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/ver7_simple_image_recon/train_epoch004/denoiser_results.csv')
df_results['ssim'].mean()
df_results['psnr'].mean()

