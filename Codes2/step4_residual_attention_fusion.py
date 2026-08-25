import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# =====================================================
# Device Setup (Mac MPS Compatible)
# =====================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# =====================================================
# LOAD CSV FILES
# =====================================================
# swin_df = pd.read_csv("swin_features.csv")
# lesion_df = pd.read_csv("lesion_features.csv")
# label_df = pd.read_csv("labels.csv")

# df = swin_df.merge(lesion_df, on="image_name")
# df = df.merge(label_df, on="image_name")

# This file is already created
df = pd.read_csv("merged_features_labels.csv")

# Extract features
swin_features = df.iloc[:, 0:768].values.astype(np.float32)
lesion_features = df.iloc[:, 769:797].values.astype(np.float32)
labels = df["diagnosis"].values.astype(np.int64)

# =====================================================
# NORMALIZE LESION FEATURES
# =====================================================
scaler = StandardScaler()
lesion_features = scaler.fit_transform(lesion_features)
joblib.dump(scaler, "lesion_scaler_attention.save")

# =====================================================
# TRAIN / VAL SPLIT
# =====================================================
X_swin_train, X_swin_val, \
X_lesion_train, X_lesion_val, \
y_train, y_val = train_test_split(
    swin_features,
    lesion_features,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# =====================================================
# DATASET
# =====================================================
class FusionDataset(Dataset):
    def __init__(self, swin_feat, lesion_feat, labels):
        self.swin_feat = torch.tensor(swin_feat, dtype=torch.float32)
        self.lesion_feat = torch.tensor(lesion_feat, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.swin_feat[idx],
            self.lesion_feat[idx],
            self.labels[idx]
        )

train_loader = DataLoader(
    FusionDataset(X_swin_train, X_lesion_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    FusionDataset(X_swin_val, X_lesion_val, y_val),
    batch_size=32,
    shuffle=False
)

# =====================================================
# RESIDUAL ATTENTION FUSION MODEL
# =====================================================
class ResidualAttentionFusionModel(nn.Module):
    def __init__(self, global_dim=768, lesion_dim=28, num_classes=5):
        super().__init__()

        self.attention_net = nn.Sequential(
            nn.Linear(lesion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, global_dim),
            nn.Sigmoid()
        )

        fusion_dim = global_dim + lesion_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, swin_feat, lesion_feat):

        alpha = self.attention_net(lesion_feat)

        # Residual gating
        weighted_swin = swin_feat * (1 + alpha)

        fused = torch.cat([weighted_swin, lesion_feat], dim=1)

        return self.classifier(fused)


model = ResidualAttentionFusionModel().to(device)

# =====================================================
# LOSS + OPTIMIZER
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# =====================================================
# TRAINING LOOP WITH SAVING
# =====================================================
epochs = 40
best_val_acc = 0.0

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):

    # ---------------- TRAIN ----------------
    model.train()
    train_correct = 0
    train_total = 0

    for swin_feat, lesion_feat, labels in train_loader:
        swin_feat = swin_feat.to(device)
        lesion_feat = lesion_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(swin_feat, lesion_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # ---------------- VALIDATION ----------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for swin_feat, lesion_feat, labels in val_loader:
            swin_feat = swin_feat.to(device)
            lesion_feat = lesion_feat.to(device)
            labels = labels.to(device)

            outputs = model(swin_feat, lesion_feat)
            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Acc: {train_acc:.4f} "
          f"Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_attention_fusion_model.pth")
        print("✅ Saved Best Attention Model")

print("Training Complete.")
print("Best Validation Accuracy:", best_val_acc)
