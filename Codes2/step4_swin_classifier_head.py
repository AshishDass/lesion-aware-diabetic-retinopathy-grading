import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# =====================================================
# Device Setup
# =====================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("merged_features_labels.csv")

# Extract features
swin_features = df.iloc[:, 0:768].values.astype(np.float32)
lesion_features = df.iloc[:, 769:797].values.astype(np.float32)
labels = df["diagnosis"].values.astype(np.int64)

# Normalize lesion features
scaler = StandardScaler()
lesion_features = scaler.fit_transform(lesion_features)
joblib.dump(scaler, "lesion_scaler_heads.save")

# Train/Val split
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
# HEAD DEFINITIONS
# =====================================================
class Head_Baseline(nn.Module):
    def __init__(self, input_dim=796, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Head_Residual(nn.Module):
    def __init__(self, input_dim=796, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, input_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        residual = x
        out = torch.relu(self.fc1(x))
        out = self.fc2(out)
        x = residual + out
        return self.classifier(x)

class Head_Bottleneck(nn.Module):
    def __init__(self, input_dim=796, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# =====================================================
# TRAIN FUNCTION
# =====================================================
def train_model(model, head_name, epochs=20):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

    best_val_acc = 0.0
    os.makedirs("checkpoints_heads", exist_ok=True)

    for epoch in range(epochs):

        # ---------------- TRAIN ----------------
        model.train()
        train_preds = []
        train_labels = []

        for swin_feat, lesion_feat, labels in train_loader:
            swin_feat = swin_feat.to(device)
            lesion_feat = lesion_feat.to(device)
            labels = labels.to(device)

            fused = torch.cat([swin_feat, lesion_feat], dim=1)

            optimizer.zero_grad()
            outputs = model(fused)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_preds = []
        val_probs = []
        val_labels_list = []

        with torch.no_grad():
            for swin_feat, lesion_feat, labels in val_loader:
                swin_feat = swin_feat.to(device)
                lesion_feat = lesion_feat.to(device)
                labels = labels.to(device)

                fused = torch.cat([swin_feat, lesion_feat], dim=1)
                outputs = model(fused)

                probs = torch.softmax(outputs, dim=1)

                val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels_list, val_preds)
        macro_f1 = f1_score(val_labels_list, val_preds, average="macro")
        auc = roc_auc_score(val_labels_list, val_probs, multi_class="ovr")

        print(f"{head_name} | Epoch {epoch+1} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"MacroF1: {macro_f1:.4f} | "
              f"AUC: {auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                f"checkpoints_heads/best_{head_name}.pth"
            )
            print(f"✅ Saved Best {head_name}")

    return best_val_acc

# =====================================================
# RUN ALL HEADS
# =====================================================
results = {}

print("\nTraining H1 - Baseline")
results["H1_Baseline"] = train_model(Head_Baseline(), "H1_Baseline")

print("\nTraining H2 - Residual")
results["H2_Residual"] = train_model(Head_Residual(), "H2_Residual")

print("\nTraining H3 - Bottleneck")
results["H3_Bottleneck"] = train_model(Head_Bottleneck(), "H3_Bottleneck")

print("\nFinal Best Validation Accuracy Comparison:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
