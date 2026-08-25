import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================

# Device selection (GPU if available, otherwise CPU)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset paths
IMAGE_DIR = "/Users/adass/Research/archive/train_images"
CSV_PATH = "/Users/adass/Research/archive/train.csv"

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_CLASSES = 5
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
NUM_FOLDS = 5

# Input image size for Swin
IMAGE_SIZE = 384


# ==============================
# DATASET CLASS
# ==============================

class APTOSDataset(Dataset):
    """
    Custom PyTorch Dataset for APTOS.
    Reads image file using id_code and corresponding diagnosis label.
    Applies transformations if provided.
    """
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # Total number of samples
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads one image-label pair.
        id_code corresponds to image filename.
        diagnosis corresponds to ICDR severity grade (0–4).
        """
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["id_code"] + ".png")
        image = Image.open(img_path).convert("RGB")
        label = row["diagnosis"]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================
# TRANSFORMATIONS
# ==============================

# Training transformations include augmentation
# These are mild retina-safe augmentations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation transformations (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ==============================
# MODEL SETUP
# ==============================

def get_model():
    """
    Loads pretrained Swin-Tiny model from timm.
    Replaces classification head to output 5 classes.
    Freezes early transformer stages to prevent overfitting.
    """
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=NUM_CLASSES,
        img_size=384
    )

    # Freeze early layers (Stage 1 and Stage 2)
    # These layers capture low-level generic features (edges, textures)
    # Freezing helps preserve pretrained knowledge and reduce overfitting
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.1" in name:
            param.requires_grad = False

    return model


# ==============================
# TRAINING FUNCTION
# ==============================

def train_one_epoch(model, loader, criterion, optimizer):
    """
    Runs one training epoch.
    Forward pass → loss computation → backward pass → parameter update.
    """
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)  # Forward pass through Swin
        loss = criterion(outputs, labels)  # Cross-entropy loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update trainable weights

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader):
    """
    Runs validation.
    Computes Quadratic Weighted Kappa (QWK),
    which is the standard metric for DR grading.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)

            # Convert logits to predicted class index
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # QWK penalizes large grading disagreements more heavily
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return qwk


# ==============================
# MAIN TRAINING LOOP (5-FOLD CV)
# ==============================

def run_training():
    """
    Performs Stratified 5-Fold Cross Validation.
    Ensures class balance in each fold.
    Saves best model per fold based on QWK.
    """
    df = pd.read_csv(CSV_PATH)

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["diagnosis"])):
        print(f"\n========== Fold {fold+1} ==========")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = APTOSDataset(train_df, IMAGE_DIR, train_transform)
        val_dataset = APTOSDataset(val_df, IMAGE_DIR, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = get_model().to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        # Optimizer updates only parameters where requires_grad=True
        # Since early layers are frozen, only later transformer blocks update
        optimizer = optim.AdamW([
            {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": LR_BACKBONE},
        ], weight_decay=0.01)

        best_qwk = 0

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            val_qwk = validate(model, val_loader)

            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | QWK: {val_qwk:.4f}")

            # Save model if performance improves
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                torch.save(model.state_dict(), f"swin_fold{fold}.pth")

        print(f"Best QWK for fold {fold}: {best_qwk:.4f}")


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(model_path, output_csv):
    """
    Loads trained Swin backbone,
    removes classification head (num_classes=0),
    extracts 768-dimensional global embeddings,
    and saves them as CSV.
    """
    df = pd.read_csv(CSV_PATH)
    dataset = APTOSDataset(df, IMAGE_DIR, val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # num_classes=0 removes classification head
    # Model outputs final pooled embedding directly
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=0,
        img_size=384
    )

    # strict=False allows ignoring missing classification head weights
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(DEVICE)
    model.eval()

    features = []

    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.to(DEVICE)

            # Forward pass now returns 768-d feature vector
            embeddings = model(images)
            features.append(embeddings.cpu().numpy())

    features = np.vstack(features)

    # Save embeddings with corresponding id_code
    feature_df = pd.DataFrame(features)
    feature_df["id_code"] = df["id_code"]
    feature_df.to_csv(output_csv, index=False)

    print("Feature extraction complete.")


if __name__ == "__main__":
    # run_training()
    extract_features("swin_fold0.pth", "swin_features.csv")


# import pandas as pd

# df = pd.read_csv("swin_features.csv")
# df["id_code"].to_csv("valid_id_codes.txt", index=False, header=False)