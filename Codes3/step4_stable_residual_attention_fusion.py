import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_curve, precision_recall_curve, auc
)

# =====================================================
# DEVICE SETUP
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

RESULTS_DIR = "results_stable_residual_attention"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("/Users/adass/Research/Codes3/merged_features_labels.csv")

swin_features = df.iloc[:, 0:768].values.astype(np.float32)
lesion_features = df.iloc[:, 769:797].values.astype(np.float32)
labels = df["diagnosis"].values.astype(np.int64)

# =====================================================
# NORMALIZE LESION FEATURES
# =====================================================
scaler = StandardScaler()
lesion_features = scaler.fit_transform(lesion_features)
joblib.dump(scaler, f"{RESULTS_DIR}/lesion_scaler.save")

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
# STABLE RESIDUAL ATTENTION MODEL
# =====================================================
class StableResidualAttentionFusion(nn.Module):
    def __init__(self, global_dim=768, lesion_dim=28, num_classes=5):
        super().__init__()

        # Reduced-capacity attention network
        self.attention_net = nn.Sequential(
            nn.Linear(lesion_dim, 128),
            nn.ReLU(),
            nn.Linear(128, global_dim),
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
        weighted_swin = swin_feat * (1 + alpha)
        fused = torch.cat([weighted_swin, lesion_feat], dim=1)
        return self.classifier(fused)

model = StableResidualAttentionFusion().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

# =====================================================
# METRIC FUNCTION
# =====================================================
def compute_metrics(labels, preds, probs):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "auc": roc_auc_score(labels, probs, multi_class="ovr"),
        "mcc": matthews_corrcoef(labels, preds),
        "kappa": cohen_kappa_score(labels, preds)
    }

# =====================================================
# TRAINING WITH EARLY STOPPING
# =====================================================
epochs = 30
patience = 5
best_f1 = 0
early_stop_counter = 0
epoch_metrics = []

for epoch in range(epochs):

    # ---------- TRAIN ----------
    model.train()
    train_preds, train_labels_list, train_probs = [], [], []

    for swin_feat, lesion_feat, labels_batch in train_loader:
        swin_feat = swin_feat.to(device)
        lesion_feat = lesion_feat.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(swin_feat, lesion_feat)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels_list.extend(labels_batch.cpu().numpy())
        train_probs.extend(probs.detach().cpu().numpy())

    train_metrics = compute_metrics(train_labels_list, train_preds, train_probs)

    # ---------- VALIDATION ----------
    model.eval()
    val_preds, val_labels_list, val_probs = [], [], []

    with torch.no_grad():
        for swin_feat, lesion_feat, labels_batch in val_loader:
            swin_feat = swin_feat.to(device)
            lesion_feat = lesion_feat.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(swin_feat, lesion_feat)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels_batch.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    val_metrics = compute_metrics(val_labels_list, val_preds, val_probs)

    print(f"\nEpoch {epoch+1}")
    print(f"Train F1: {train_metrics['f1']:.4f} | "
          f"Val F1: {val_metrics['f1']:.4f}")

    epoch_metrics.append({
        "epoch": epoch+1,
        **{f"train_{k}":v for k,v in train_metrics.items()},
        **{f"val_{k}":v for k,v in val_metrics.items()}
    })

    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   f"{RESULTS_DIR}/best_model.pth")
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

pd.DataFrame(epoch_metrics).to_csv(
    f"{RESULTS_DIR}/epoch_metrics.csv",
    index=False
)

# =====================================================
# FINAL EVALUATION
# =====================================================
val_labels_array = np.array(val_labels_list)
val_probs_array = np.array(val_probs)
val_preds_array = np.array(val_preds)

final_metrics = compute_metrics(
    val_labels_array,
    val_preds_array,
    val_probs_array
)

with open(f"{RESULTS_DIR}/final_report_metrics.txt", "w") as f:
    for k, v in final_metrics.items():
        f.write(f"{k.upper()}: {v:.6f}\n")

pd.DataFrame([final_metrics]).to_csv(
    f"{RESULTS_DIR}/final_report_metrics.csv",
    index=False
)

# =====================================================
# ROC + PR
# =====================================================
val_labels_bin = label_binarize(val_labels_array, classes=list(range(5)))

plt.figure(figsize=(8,6))
for i in range(5):
    fpr, tpr, _ = roc_curve(val_labels_bin[:, i], val_probs_array[:, i])
    plt.plot(fpr, tpr,
             label=f"Class {i} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.title("ROC - Stable Residual Attention")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ROC.png")
plt.close()

plt.figure(figsize=(8,6))
for i in range(5):
    precision_c, recall_c, _ = precision_recall_curve(
        val_labels_bin[:, i],
        val_probs_array[:, i]
    )
    plt.plot(recall_c, precision_c,
             label=f"Class {i} (AUC={auc(recall_c,precision_c):.3f})")
plt.title("PR - Stable Residual Attention")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/PR.png")
plt.close()

# =====================================================
# SAVE CONFIDENCE SCORES
# =====================================================
results_df = pd.DataFrame({
    "true_label": val_labels_array,
    "predicted_label": val_preds_array,
    "confidence": np.max(val_probs_array, axis=1)
})

for i in range(5):
    results_df[f"class_{i}_prob"] = val_probs_array[:, i]

results_df.to_csv(
    f"{RESULTS_DIR}/val_predictions_with_confidence.csv",
    index=False
)

print("\nStable Residual Attention Fusion complete.")
print("Artifacts saved in:", RESULTS_DIR)
