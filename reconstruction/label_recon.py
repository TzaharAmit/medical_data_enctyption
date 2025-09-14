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
# --- Robust classifier for label reconstruction from adversarial inputs ---
class AdvLabelReconstructor(nn.Module):
    """ResNet50 -> single logit (binary). Unfreezes from layer4 by default."""
    def __init__(self, unfreeze_from="layer4", pretrained=True, dropout=0.0):
        super().__init__()
        m = models.resnet50(pretrained=pretrained)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, 1)  # single logit
        )
        self.backbone = m

        # freeze all, then unfreeze head + last block
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.fc.parameters():
            p.requires_grad = True
        if unfreeze_from is not None:
            turn_on = False
            for name, p in self.backbone.named_parameters():
                if name.startswith(unfreeze_from):
                    turn_on = True
                if turn_on:
                    p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)  # (B,1) logits


#%% read data:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adv_train_df = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/df_results.csv')
image_root = r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data'   
adv_train_df['adv_path'] = 'reconstruction/square/train/' + adv_train_df['image'].str.lstrip("./")
adv_train_df['clean_path'] = 'nih/' + adv_train_df['image'].str.lstrip("./") 

adv_test_df = pd.read_csv(r'//home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/df_test_results.csv')
image_root = r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data'   
adv_test_df['adv_path'] = 'reconstruction/square/train/' + adv_test_df['image'].str.lstrip("./")
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


def model_train(model, epoch, train_loader, optimizer, ckpt_dir, device=device, pos_weight=None):
    """
    Training loop for AdvLabelReconstructor.
    Trains on adversarial inputs (x_adv) to predict clean labels (y).
    """
    model.train()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for x_adv, _, y in train_loader:
        x_adv = x_adv.to(device)
        y     = y.to(device).float().unsqueeze(1)  # (B,1)

        logits = model(imagenet_normalize(x_adv))
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (logits.sigmoid() >= 0.5).float()
            acc   = (preds == y).float().mean().item()

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_acc  = total_acc / max(1, n_batches)
    torch.save(
    {"epoch": int(epoch), "split": "train",
     "loss": avg_loss, "acc": avg_acc},
    f"{ckpt_dir}/train_metrics_epoch{epoch:03}.pt")

    torch.save(model.state_dict(), f"{ckpt_dir}/train_epoch{epoch:03d}.pt")

    print(f"[Train {epoch}] loss={avg_loss:.4f} acc={avg_acc*100:.1f}%")
    return {"loss": avg_loss, "acc": avg_acc}




@torch.no_grad()
def model_test(model, epoch, test_loader, ckpt_dir, device=device, pos_weight=None):
    """
    Eval loop for AdvLabelReconstructor.
    Returns a dict with loss & acc (and optionally prec/rec/f1/auroc).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_probs, all_labels = [], []

    for x_adv, _, y in test_loader:
        x_adv = x_adv.to(device)
        y     = y.to(device).float().unsqueeze(1)  # (B,1)

        logits = model(imagenet_normalize(x_adv))
        loss   = criterion(logits, y)

        probs = logits.sigmoid()
        preds = (probs >= 0.5).float()
        acc   = (preds == y).float().mean().item()

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_acc  = total_acc / max(1, n_batches)
    metrics = {"loss": avg_loss, "acc": avg_acc}

    # Save a light checkpoint/metrics file for parity with train side (optional)
    torch.save(
        {"epoch": int(epoch), "split": "val", **metrics},
        f"{ckpt_dir}/val_metrics_epoch{epoch:03d}.pt"
    )
    torch.save(model.state_dict(), f"{ckpt_dir}/val_epoch{epoch:03d}.pt")
    print(f"[Val  {epoch}] loss={metrics['loss']:.4f} acc={metrics['acc']*100:.1f}%")
    return metrics


#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Model
model = AdvLabelReconstructor(
    unfreeze_from="layer4",  # train last ResNet block + classifier
    pretrained=True,
    dropout=0.1).to(device)

# 2) Optimizer (only params that require grad)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=1e-4)

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver6_simple_label_recon/"

from tqdm import tqdm
EPOCHS = 50
with tqdm(total=EPOCHS) as pbar:
    for epoch in range(1, EPOCHS + 1):
        train_stats = model_train(model, epoch, train_dataloader, optimizer, ckpt_dir, device=device, pos_weight=None)
        val_stats   = model_test(model, epoch, test_dataloader, ckpt_dir, device=device, pos_weight=None)
        pbar.update(1)


        
      
#%% run selected model on test data:
from pathlib import Path

@torch.no_grad()
def model_verify(model, test_loader, adv_test_df, ckpt_dir, device):
    """
    Run AdvLabelReconstructor on the test set and record predictions.
    """
    model.eval()
    model.to(device)

    records = []
    idx = 0

    for x_adv, _, y in test_loader:
        bsz = y.size(0)
        batch_df = adv_test_df.iloc[idx: idx + bsz]

        x_adv = x_adv.to(device)
        logits = model(imagenet_normalize(x_adv))           # (B,1)
        probs  = torch.sigmoid(logits).squeeze(1).cpu()     # (B,)
        preds  = (probs >= 0.5).long().cpu()                # (B,)

        # collect rows
        for i in range(bsz):
            row = {
                "adv_path":            batch_df.iloc[i]["adv_path"],
                "clean_path":          batch_df.iloc[i]["clean_path"],
                "true_label":          int(batch_df.iloc[i]["true_label"]),
                "pred_after_attack":   int(batch_df.iloc[i]["pred_after_attack"]),
                "pred_after_recon":    int(preds[i].item()),
                "prob1_after_recon":   float(probs[i].item()),
            }
            records.append(row)
        idx += bsz

    df_results = pd.DataFrame(records)
    out_csv = os.path.join(ckpt_dir, "reconstruction_label_results.csv")
    df_results.to_csv(out_csv, index=False)
    print(df_results.head())
    return df_results


# Set path and device
selected_ckpt = Path(r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver6_simple_label_recon/train_epoch002.pt")

# Load model and optimizer
model = AdvLabelReconstructor(unfreeze_from="layer4", pretrained=True, dropout=0.1).to(device)
model.load_state_dict(torch.load(selected_ckpt, map_location=device))

ckpt_dir="/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/all_results/ver6_simple_label_recon/train_epoch002"

df_results = model_verify(model, test_dataloader, adv_test_df, ckpt_dir, device)


#%% get test results: label recon:
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

#df_results = pd.read_csv(r'/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/all_results/ver6_simple_label_recon/train_epoch004/reconstruction_label_results.csv')
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


#%% results visualization:
import re, glob
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/square/models/ver6_simple_label_recon/"

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

#%% visialize test results:

#%% images visualization:
# --- Load checkpoint you choose ---
selected_ckpt_dir = r"/home/linuxu/Desktop/final_project/Adversarial-Data-Encryption/data/reconstruction/fgsm/models/ver6_simple_label_recon/train_epoch004.pt"
model = AdvLabelReconstructor(
    unfreeze_from="layer4",  # train last ResNet block + classifier
    pretrained=True,
    dropout=0.1).to(device)
model.load_state_dict(torch.load(selected_ckpt_dir, map_location=device))
model.eval()

# --- Take first few rows from adv_test_df ---
# choose a few rows
subset = adv_test_df.head(8)  # change N as you like

for idx, row in subset.iterrows():
    # --- Load tensors (no normalization here; for display keep [0,1]) ---
    x_adv = adv_test_dataset._load_xray_as_tensor(
        os.path.join(image_root, row["adv_path"])
    ).unsqueeze(0).to(device)  # (1,3,224,224)

    x_clean = adv_test_dataset._load_xray_as_tensor(
        os.path.join(image_root, row["clean_path"])
    ).unsqueeze(0).to(device)

    y_true      = int(row["true_label"])
    y_attacked  = int(row.get("pred_after_attack", -1))  # if column exists

    # --- Forward pass (normalize only for the model input) ---
    with torch.no_grad():
        logit = model(imagenet_normalize(x_adv))        # (1,1)
        p1 = torch.sigmoid(logit).item()                # prob of class 1
        y_recon = int(p1 >= 0.5)                        # reconstructed label

    # --- Plot CLEAN vs ADV (you can add a 3rd panel if you want diff/heatmap) ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    imgs = [x_clean[0], x_adv[0]]
    titles = [
        f"CLEAN (true={y_true})",
        f"ADV (atk={y_attacked if y_attacked!=-1 else 'NA'}, recon={y_recon}, p1={p1:.2f})"
    ]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im.permute(1,2,0).cpu().clamp(0,1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
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
