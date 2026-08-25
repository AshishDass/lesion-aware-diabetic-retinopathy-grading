import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Swin feature CSV (should contain 768 features + image_name)
swin_df = pd.read_csv("swin_features.csv")

# Lesion feature CSV (28 features + image_name)
lesion_df = pd.read_csv("lesion_features_val_id.csv")
lesion_df = lesion_df.drop(columns=["flag_mild_DR", "flag_severe_NPDR", "flag_vision_threat"])
# lesion_df.to_csv("lesion_features_val.csv", index=False)
print("Dropped specified columns and saved the file.")

# Label CSV (image_name + label 0-4)
label_df = pd.read_csv("labels.csv")

# Merge all
df = swin_df.merge(lesion_df, on="id_code")
df = df.merge(label_df, on="id_code")
df.to_csv("merged_features_labels.csv", index=False)
# print(df.columns.tolist())
# print("Merged dataset shape:", df.shape)
# print(df.head())

# =====================================================
# SPLIT FEATURES
# =====================================================

# Assuming first column = image_name
# Swin features = next 768 columns
# Lesion features = next 28 columns
# Label column = "diagnosis"

swin_features = df.iloc[:, 0:768].values.astype(np.float32)
lesion_features = df.iloc[:, 769:797].values.astype(np.float32)
labels = df["diagnosis"].values.astype(np.int64)

# Column 768 is id_code
#### For testing purposes, we can print the shapes of the features and labels
# print(labels)
# print(swin_features)
# print(lesion_features)





# =====================================================
# NORMALIZE LESION FEATURES (IMPORTANT)
# =====================================================
scaler = StandardScaler()
lesion_features = scaler.fit_transform(lesion_features)

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

train_dataset = FusionDataset(X_swin_train, X_lesion_train, y_train)
val_dataset = FusionDataset(X_swin_val, X_lesion_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =====================================================
# EARLY FUSION MODEL
# =====================================================
class EarlyFusionModel(nn.Module):
    def __init__(self, global_dim=768, lesion_dim=28, num_classes=5):
        super(EarlyFusionModel, self).__init__()

        fusion_dim = global_dim + lesion_dim  # 796

        self.model = nn.Sequential(
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
        fused = torch.cat([swin_feat, lesion_feat], dim=1)
        return self.model(fused)

model = EarlyFusionModel().to(device)

# =====================================================
# LOSS + OPTIMIZER
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# =====================================================
# TRAINING LOOP
# =====================================================
epochs = 30
best_val_acc = 0.0

for epoch in range(epochs):

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for swin_feat, lesion_feat, labels in train_loader:
        swin_feat = swin_feat.to(device)
        lesion_feat = lesion_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(swin_feat, lesion_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

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
    
    # ---------------- SAVE BEST MODEL ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("checkpoints", exist_ok=True)  # Ensure directory exists
        torch.save(model.state_dict(), "checkpoints/best_early_fusion_model.pth")
        print("✅ Saved Best Model")


print("Training Complete.")
print("Best Validation Accuracy:", best_val_acc)