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

# model class:
class VAEClassifier(nn.Module):
    def __init__(self, latent_dim=256, freeze_encoder=True):
        super().__init__()

        # --- Encoder (pretrained, frozen) ---
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # (B,2048,1,1)
        if freeze_encoder:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
        feat_dim = resnet.fc.in_features  # 2048

        # --- Latent space ---
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

        # --- Decoder (unchanged) ---
        self.decoder_fc = nn.Linear(latent_dim, 256*7*7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),     # 224x224
            nn.Sigmoid()
        )

        # --- Classifier: deeper MLP, single logit ---
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)   # one logit for binary classification
        )

    def encode(self, x):
        h = self.feature_extractor(x).flatten(1)  # (B,2048)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 256, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        logit = self.classifier(z)   # shape (B,1)
        return x_recon, logit, mu, logvar

#%% read data:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adv_train_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/weak_fgsm/train_eps_0.02/df_results.csv')
image_root = r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data'   
adv_train_df['adv_path'] = 'reconstruction/fgsm/weak_fgsm/train_eps_0.02/' + adv_train_df['image'].str.lstrip("./")
adv_train_df['clean_path'] = 'nih/' + adv_train_df['image'].str.lstrip("./") 

adv_test_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/weak_fgsm/train_eps_0.02/df_test_results.csv')
image_root = r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data'   
adv_test_df['adv_path'] = 'reconstruction/fgsm/weak_fgsm/train_eps_0.02/' + adv_test_df['image'].str.lstrip("./")
adv_test_df['clean_path'] = 'nih/' + adv_test_df['image'].str.lstrip("./") 

adv_train_dataset = XRayAdvDataset(adv_train_df, image_root, 'adv_path', 'clean_path', 'true_label')
adv_test_dataset = XRayAdvDataset(adv_test_df, image_root, 'adv_path', 'clean_path', 'true_label')

batch_size = 16
train_dataloader = DataLoader(adv_train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(adv_test_dataset, batch_size=batch_size, shuffle=False)

# model
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[..., None, None]
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])[..., None, None]

def imagenet_normalize(x):   # x in [0,1], (B,3,H,W)
    return (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)

def build_optimizer(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=1e-3)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        np.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.float().unsqueeze(0).unsqueeze(0)
    return window.expand(channel, 1, window_size, window_size).contiguous()

def ssim(img1, img2, window_size=11, val_range=1.0, size_average=True):
    # Assumes inputs are normalized to [0, 1]
    (_, channel, height, width) = img1.size()
    real_size = min(window_size, height, width)
    window = create_window(real_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=0, groups=channel)
    mu2 = F.conv2d(img2, window, padding=0, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - mu1_mu2

    C1 = (0.01 * val_range)**2
    C2 = (0.03 * val_range)**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def model_train(model, epoch, train_loader, optimizer, lam, beta, ckpt_dir):
    model.train()
    totals = {"loss":0,"recon":0,"ce":0,"acc":0}
    for x_adv, x_clean, y in train_loader:
        x_adv, x_clean, y = x_adv.cuda(), x_clean.cuda(), y.cuda()
        x_adv_norm = imagenet_normalize(x_adv)

        x_recon, logit, mu, logvar = model(x_adv_norm)
        recon = 1 - ssim(x_recon, x_clean)  # SSIM in [0,1], so 1 - SSIM is the loss
        # BCE-with-logits expects float targets in {0,1} and shape (B,1)
        y_float = y.float().unsqueeze(1)
        ce = F.binary_cross_entropy_with_logits(logit, y_float)  # keep key name 'ce' for compatibility        loss  = recon + kl + lam * ce
        loss  = recon + lam * ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accuracy via sigmoid threshold 0.5
        preds = (logit.sigmoid() >= 0.5).long().squeeze(1)  # (B,)
        acc   = (preds == y).float().mean().item()        
        totals["loss"]+=loss.item(); totals["recon"]+=recon.item()
        totals["ce"]+=ce.item(); totals["acc"]+=acc

    n = len(train_loader)
    avg = {k:v/n for k,v in totals.items()}
    torch.save(
    {"epoch": int(epoch), "split": "train",
     "loss": float(avg["loss"]), "recon": float(avg["recon"]),
     "ce": float(avg["ce"]), "acc": float(avg["acc"])},
    f"{ckpt_dir}/train_metrics_epoch{epoch:03}.pt")
    torch.save(model.state_dict(), f"{ckpt_dir}/train_epoch{epoch}.pt")
    print(f"[Train {epoch}] loss={avg['loss']:.4f} recon={avg['recon']:.4f} "
          f"ce={avg['ce']:.4f} acc={avg['acc']*100:.1f}%")
    return avg


def model_test(model, epoch, test_loader, lam, beta, ckpt_dir, set_type="val"):
    model.eval()
    totals = {"loss": 0, "recon": 0, "ce": 0, "acc": 0}
    with torch.no_grad():
        for x_adv, x_clean, y in test_loader:
            x_adv, x_clean, y = x_adv.cuda(), x_clean.cuda(), y.cuda()
            x_adv_norm = imagenet_normalize(x_adv)
            # single-logit head
            x_recon, logit, mu, logvar = model(x_adv_norm)   # logit: (B,1)
            # losses
            recon = 1 - ssim(x_recon, x_clean)
            # BCE-with-logits expects float targets in {0,1} with shape (B,1)
            y_float = y.float().unsqueeze(1)
            ce      = F.binary_cross_entropy_with_logits(logit, y_float)
            loss = recon + lam * ce
            # accuracy: sigmoid + 0.5 threshold → {0,1}
            preds = (logit.sigmoid() >= 0.5).long().squeeze(1)  # (B,)
            acc   = (preds == y).float().mean().item()

            totals["loss"] += loss.item()
            totals["recon"] += recon.item()
            totals["ce"] += ce.item()
            totals["acc"] += acc

    n = len(test_loader)
    avg = {k: v / n for k, v in totals.items()}
    torch.save(
        {"epoch": int(epoch), "split": "val",
         "loss": float(avg["loss"]), "recon": float(avg["recon"]),
         "ce": float(avg["ce"]), "acc": float(avg["acc"])},
        f"{ckpt_dir}/val_metrics_epoch{epoch:03}.pt"
    )
    torch.save(model.state_dict(), f"{ckpt_dir}/{set_type}_epoch{epoch}.pt")
    print(f"[{set_type} {epoch}] loss={avg['loss']:.4f} recon={avg['recon']:.4f} "
          f"ce={avg['ce']:.4f} acc={avg['acc']*100:.1f}%")
    return avg

#%%

model = VAEClassifier().cuda()
# train_epochs(model, train_dataloader, 'train', epochs=15, lam=1.0)
ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver_5_no_kl_w_ssim_weak_eps_0.02/"
optimizer = build_optimizer(model)  

with tqdm(total=50) as pbar:
    for epoch in range(1, 51):
        model_train(model, epoch, train_dataloader, optimizer, lam=1.0, beta=1.0, ckpt_dir=ckpt_dir)
        model_test(model, epoch, test_dataloader, lam=1.0, beta=1.0, ckpt_dir=ckpt_dir, set_type="val")
        pbar.update(1)



        
      
#%% run selected model on test data:
from pathlib import Path

# def model_verify(model, epoch, test_loader, lam, beta, ckpt_dir, set_type="val"):
#     all_results = pd.DataFrame()
#     model.eval()
#     model.to(device)
    
#     totals = {"loss": 0, "recon": 0, "kl": 0, "ce": 0, "acc": 0}
#     with torch.no_grad():
#         for x_adv, x_clean, y in test_loader:
#             x_adv, x_clean, y = x_adv.cuda(), x_clean.cuda(), y.cuda()
#             x_adv_norm = imagenet_normalize(x_adv)
#             # single-logit head
#             x_recon, logit, mu, logvar = model(x_adv_norm)   # logit: (B,1)
#             # losses
#             recon = F.mse_loss(x_recon, x_clean)
#             kl    = kl_divergence(mu, logvar, "mean") * beta
#             # BCE-with-logits expects float targets in {0,1} with shape (B,1)
#             y_float = y.float().unsqueeze(1)
#             ce      = F.binary_cross_entropy_with_logits(logit, y_float)
#             loss = recon*5 + kl + lam * ce
#             # accuracy: sigmoid + 0.5 threshold → {0,1}
#             preds = (logit.sigmoid() >= 0.5).long().squeeze(1)  # (B,)
#             acc   = (preds == y).float().mean().item()
                        
#     print(f"[{set_type} {epoch}] loss={avg['loss']:.4f} recon={avg['recon']:.4f} "
#           f"kl={avg['kl']:.4f} ce={avg['ce']:.4f} acc={avg['acc']*100:.1f}%")

def model_verify(model, test_loader, adv_test_df, ckpt_dir, device):
    model.eval()
    model.to(device)
    records = []

    with torch.no_grad():
        idx = 0
        for x_adv, x_clean, y in test_loader:
            batch_size = y.size(0)
            # Matching batch rows in original DataFrame
            batch_df = adv_test_df.iloc[idx:idx+batch_size]
            x_adv = x_adv.to(device)
            x_adv_norm = imagenet_normalize(x_adv)
            # Model prediction
            x_recon, logit, mu, logvar = model(x_adv_norm)
            preds = (logit.sigmoid() >= 0.5).long().squeeze(1)  # (B,)
            # Record results
            for i in range(batch_size):
                records.append({
                    "image_path": batch_df.iloc[i]["clean_path"],   # or adv_path
                    "true_label": int(batch_df.iloc[i]["true_label"]),
                    "pred_after_attack": int(batch_df.iloc[i]["pred_after_attack"]),
                    "pred_after_recon": int(preds[i].cpu().item()),
                })
            idx += batch_size

    df_results = pd.DataFrame(records)
    df_results.to_csv(f"{ckpt_dir}/reconstruction_label_results.csv", index=False)
    print(df_results.head())
    return df_results

# Set path and device
selected_ckpt = Path(r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver3_recon_5/train_epoch21.pt")

# Load model and optimizer
model = VAEClassifier()
model.load_state_dict(torch.load(selected_ckpt))

ckpt_dir="/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/er3_recon_5_train_epoch21/"

df_results = model_verify(model, test_dataloader, adv_test_df, ckpt_dir, device)

#%% get test results: label recon:
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

df_results = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/er3_recon_5_train_epoch21/reconstruction_label_results.csv')
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


#%% results visualization:
import re, glob
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver_5_no_kl_w_ssim_weak_eps_0.02/"

# ------- helpers -------
epoch_num = lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1))

def files(pattern):
    fs = glob.glob(os.path.join(ckpt_dir, pattern))
    return [p for _, p in sorted((epoch_num(p), p) for p in fs)]

def load_metrics(path):
    # CPU deserialization avoids GPU overhead; only grab the 'loss' field
    obj = torch.load(path, map_location="cpu")
    return float(obj["loss"]), float(obj["recon"]), float(obj["ce"])

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

train_losses, train_recon, train_ce= zip(*train_metrics)
val_losses, val_recon, val_ce = zip(*val_metrics)


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
plt.plot(range(1, len(train_ce)+1), train_ce, label="Training", marker="o")
plt.plot(range(1, len(val_ce)+1),   val_ce,   label="Validation", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation CE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------- plot -------
plt.figure(figsize=(8,4))
plt.plot(range(1, len(train_ce)+1), train_recon, label="Training", marker="o")
plt.plot(range(1, len(val_ce)+1),   val_recon,   label="Validation", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation recon")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%% visialize test results:

#%% images visualization:
# --- Load checkpoint you choose ---
selected_ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver_5_no_kl_w_ssim_weak_eps_0.02/train_epoch21.pt"
model = VAEClassifier(latent_dim=256).to(device)
model.load_state_dict(torch.load(selected_ckpt_dir, map_location=device))
model.eval()

# --- Take first few rows from adv_test_df ---
subset = adv_test_df.head(4)   # show 4 samples

for idx, row in subset.iterrows():
    # Load tensors
    x_adv = adv_test_dataset._load_xray_as_tensor(
        os.path.join(image_root, row["adv_path"])
    ).unsqueeze(0).to(device)
    x_clean = adv_test_dataset._load_xray_as_tensor(
        os.path.join(image_root, row["clean_path"])
    ).unsqueeze(0).to(device)
    y_true = int(row["true_label"])
    y_attacked = int(row["pred_after_attack"])

    # Model forward
    with torch.no_grad():
        x_recon, logits, _, _ = model(imagenet_normalize(x_adv))
        y_recon = logits.argmax(1).item()

    # Plot side by side
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    imgs = [x_clean[0], x_adv[0], x_recon[0]]
    titles = [
        f"CLEAN\ntrue label={y_true}",
        f"ADV\nattacked label={y_attacked}",
        f"RECON\nreconstructed label={y_recon}"
    ]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im.permute(1,2,0).cpu().clamp(0,1))
        ax.set_title(title)
        ax.axis("off")
    plt.show()


#%% test tiny subset:
from torch.utils.data import Subset, DataLoader
import os

# 1) tiny subset
N_TINY = 16
train_tiny = Subset(adv_train_dataset, list(range(N_TINY)))
val_tiny   = Subset(adv_test_dataset,  list(range(min(16, len(adv_test_dataset)))))

# 2) loaders: use full tiny set as a single batch (fast memorization)
train_dataloader = DataLoader(train_tiny, batch_size=N_TINY, shuffle=True, num_workers=0, pin_memory=True)
#test_dataloader  = DataLoader(val_tiny,   batch_size=N_TINY, shuffle=False, num_workers=0, pin_memory=True)
test_dataloader = train_dataloader
# 3) model + optimizer (keep your 2-LR optimizer)

model = VAEClassifier(latent_dim=64).cuda()
optimizer = build_optimizer(model)

# 4) run a few epochs with small KL weight (beta) so it doesn’t fight memorization
EPOCHS = 25
lam, beta = 1.0, 1e-3

from tqdm import tqdm
with tqdm(total=EPOCHS) as pbar:
    for epoch in range(1, EPOCHS + 1):
        model_train(model, epoch, train_dataloader, optimizer, lam=lam, beta=beta, ckpt_dir=ckpt_dir)
        model_test(model, epoch, test_dataloader, lam=lam, beta=beta, ckpt_dir=ckpt_dir, set_type="val")
        pbar.update(1)
