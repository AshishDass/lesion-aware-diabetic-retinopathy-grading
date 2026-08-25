import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

# =========================
# CONFIG
# =========================
DATA_ROOT = r"E:\NIT Delhi\Research 2\DDRseg"
IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LESIONS = ["MA", "HE", "EX", "SE"]
NUM_CLASSES = len(LESIONS)

# Early stopping
PATIENCE = 7

# Checkpointing
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
BEST_CKPT_PATH = os.path.join(CKPT_DIR, "best_model.pth")
LAST_CKPT_PATH = os.path.join(CKPT_DIR, "last_model.pth")

# Test predictions
TEST_SAVE_DIR = "results/test_predictions"
DEBUG_HEATMAP_DIR = "results/debug_heatmaps"
os.makedirs(TEST_SAVE_DIR, exist_ok=True)
os.makedirs(DEBUG_HEATMAP_DIR, exist_ok=True)
for l in LESIONS:
    os.makedirs(os.path.join(TEST_SAVE_DIR, l), exist_ok=True)
    os.makedirs(os.path.join(DEBUG_HEATMAP_DIR, l), exist_ok=True)

# =========================
# SEED
# =========================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# =========================
# PREPROCESS
# =========================
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)

def preprocess_mask(path):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    if os.path.exists(path):
        # Read TIFF safely (preserve bit depth)
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if m is None:
            raise ValueError(f"Failed to read mask: {path}")

        # If mask is multi-channel, take first channel
        if len(m.shape) == 3:
            m = m[:, :, 0]

        # Resize without interpolation artifacts
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Normalize binary mask (handles 255, 65535, etc.)
        m = (m > 0).astype(np.float32)

        mask[m == 1] = 1.0

    return torch.tensor(mask, dtype=torch.float32)


# =========================
# DATASET
# =========================
class DDRDataset(Dataset):
    def __init__(self, split):
        self.images = sorted(glob(f"{DATA_ROOT}/{split}/image/*"))
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        name = os.path.basename(img_path)

        img = preprocess_image(img_path)

        masks = []
        for lesion in LESIONS:
            mask_path = f"{DATA_ROOT}/{self.split}/label/{lesion}/{name}"
            masks.append(preprocess_mask(mask_path))

        masks = torch.stack(masks, dim=0)  # (4, H, W)
        return img, masks

# =========================
# MODEL (VERSION-SAFE)
# =========================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,   # multilabel channels
    activation=None
).to(DEVICE)

# =========================
# LOSS & METRICS
# =========================
# Critical: positive class weighting to fight background collapse
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=DEVICE))
dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

def dice_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5

    # If no GT pixels, ignore this sample
    if target.sum() == 0:
        return torch.tensor(float("nan"), device=pred.device)

    intersection = (pred & target).float().sum()
    union = pred.float().sum() + target.float().sum()

    return (2 * intersection + eps) / (union + eps)

# =========================
# TRAIN / VAL LOOP
# =========================
def run_epoch(loader, train=True):
    model.train() if train else model.eval()

    loss_log = []
    dice_log = {l: [] for l in LESIONS}

    with torch.set_grad_enabled(train):
        for img, masks in tqdm(loader, leave=False):
            img = img.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(img)  # (B, 4, H, W)
            loss = 0

            for i, l in enumerate(LESIONS):
                p = preds[:, i:i+1]
                t = masks[:, i:i+1]

                loss_l = 0.5 * bce(p, t) + 0.5 * dice(p, t)
                loss += loss_l

                d = dice_score(p, t)
                dice_log[l].append(d.item())

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_log.append(loss.item())

    mean_loss = np.mean(loss_log)
    mean_dice = {l: np.nanmean(dice_log[l]) for l in LESIONS}

    return mean_loss, mean_dice

# =========================
# TEST METRICS
# =========================
@torch.no_grad()
def evaluate_test_set(loader):
    model.eval()
    dice_log = {l: [] for l in LESIONS}

    for img, masks in tqdm(loader, desc="Testing"):
        img = img.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(img)

        for i, l in enumerate(LESIONS):
            d = dice_score(preds[:, i:i+1], masks[:, i:i+1])
            dice_log[l].append(d.item())

    print("\n===== TEST METRICS (HONEST) =====")
    for l in LESIONS:
        print(f"{l} Dice: {np.nanmean(dice_log[l]):.4f}")

# =========================
# SAVE TEST MASKS + HEATMAPS
# =========================
@torch.no_grad()
def save_test_predictions():
    model.eval()
    dataset = DDRDataset("test")

    for idx in tqdm(range(len(dataset)), desc="Saving test predictions"):
        img, _ = dataset[idx]
        img = img.unsqueeze(0).to(DEVICE)

        preds = model(img)
        name = os.path.basename(dataset.images[idx])

        for i, l in enumerate(LESIONS):
            prob = torch.sigmoid(preds[0, i]).cpu().numpy()

            # Save probability heatmap (debug)
            heatmap = (prob * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(DEBUG_HEATMAP_DIR, l, name), heatmap)

            # Lower threshold to avoid black masks
            mask = (prob > 0.2).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(TEST_SAVE_DIR, l, name), mask)

# =========================
# MAIN
# =========================
train_loader = DataLoader(DDRDataset("train"), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(DDRDataset("valid"), BATCH_SIZE)
test_loader  = DataLoader(DDRDataset("test"), batch_size=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_val_dice = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_dice = run_epoch(train_loader, train=True)
    val_loss, val_dice = run_epoch(val_loader, train=False)

    mean_val_dice = np.nanmean(list(val_dice.values()))

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val   Loss: {val_loss:.4f}")
    print(f"Val Mean Dice (honest): {mean_val_dice:.4f}")

    for l in LESIONS:
        print(f"{l} Dice → Train: {train_dice[l]:.4f} | Val: {val_dice[l]:.4f}")

    ckpt = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_mean_dice": mean_val_dice
    }

    torch.save(ckpt, LAST_CKPT_PATH)

    if mean_val_dice > best_val_dice:
        best_val_dice = mean_val_dice
        epochs_no_improve = 0
        torch.save(ckpt, BEST_CKPT_PATH)
        print("✅ Best model saved")
    else:
        epochs_no_improve += 1
        print(f"⏳ No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= PATIENCE:
        print("🛑 Early stopping triggered")
        break

# =========================
# TEST USING BEST MODEL
# =========================
print("\nLoading best model for testing...")
best_ckpt = torch.load(BEST_CKPT_PATH, map_location=DEVICE)
model.load_state_dict(best_ckpt["model_state"])

evaluate_test_set(test_loader)
save_test_predictions()
