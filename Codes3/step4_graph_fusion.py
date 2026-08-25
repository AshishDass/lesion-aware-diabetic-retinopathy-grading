import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

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
# DEVICE
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

RESULTS_DIR = "results_graph_multihead"
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
# TRAIN/VAL SPLIT
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
        return self.swin_feat[idx], self.lesion_feat[idx], self.labels[idx]

train_loader = DataLoader(FusionDataset(X_swin_train, X_lesion_train, y_train),
                          batch_size=32, shuffle=True)

val_loader = DataLoader(FusionDataset(X_swin_val, X_lesion_val, y_val),
                        batch_size=32, shuffle=False)

# =====================================================
# GRAPH MULTI-HEAD MODEL
# =====================================================
class GraphMultiHeadFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_nodes = 4
        self.node_dim = 7
        self.embed_dim = 128

        self.node_encoder = nn.Linear(self.node_dim, self.embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(768 + self.embed_dim),
            nn.Linear(768 + self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, swin_feat, lesion_feat, return_attention=False):

        B = lesion_feat.size(0)
        nodes = lesion_feat.view(B, self.num_nodes, self.node_dim)

        node_emb = self.node_encoder(nodes)

        attn_output, attn_weights = self.attention(
            node_emb, node_emb, node_emb,
            need_weights=True,
            average_attn_weights=False
        )

        node_emb = self.norm(node_emb + attn_output)

        graph_repr = node_emb.mean(dim=1)

        fused = torch.cat([swin_feat, graph_repr], dim=1)

        logits = self.classifier(fused)

        if return_attention:
            return logits, attn_weights
        else:
            return logits

model = GraphMultiHeadFusion().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

# =====================================================
# METRICS FUNCTION
# =====================================================
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y_true, y_prob, multi_class="ovr"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred)
    }

# =====================================================
# TRAINING LOOP
# =====================================================
epochs = 30
patience = 5
best_f1 = 0
early_counter = 0
epoch_metrics = []

for epoch in range(epochs):

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

    print(f"Epoch {epoch+1} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")

    epoch_metrics.append({
        "epoch": epoch+1,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()}
    })

    if val_metrics["f1"] > best_f1:
        best_f1 = val_metrics["f1"]
        early_counter = 0
        torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model.pth")
    else:
        early_counter += 1

    if early_counter >= patience:
        print("Early stopping triggered.")
        break

pd.DataFrame(epoch_metrics).to_csv(f"{RESULTS_DIR}/epoch_metrics.csv", index=False)

# =====================================================
# FINAL METRICS
# =====================================================
val_labels_array = np.array(val_labels_list)
val_probs_array = np.array(val_probs)
val_preds_array = np.array(val_preds)

final_metrics = compute_metrics(val_labels_array, val_preds_array, val_probs_array)

pd.DataFrame([final_metrics]).to_csv(f"{RESULTS_DIR}/final_report_metrics.csv", index=False)

with open(f"{RESULTS_DIR}/final_report_metrics.txt", "w") as f:
    for k, v in final_metrics.items():
        f.write(f"{k.upper()}: {v:.6f}\n")

# =====================================================
# ROC + PR
# =====================================================
val_labels_bin = label_binarize(val_labels_array, classes=list(range(5)))

plt.figure(figsize=(8,6))
for i in range(5):
    fpr, tpr, _ = roc_curve(val_labels_bin[:, i], val_probs_array[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.legend()
plt.title("ROC - Graph MultiHead")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ROC.png")
plt.close()

plt.figure(figsize=(8,6))
for i in range(5):
    precision_c, recall_c, _ = precision_recall_curve(
        val_labels_bin[:, i], val_probs_array[:, i])
    plt.plot(recall_c, precision_c,
             label=f"Class {i} (AUC={auc(recall_c,precision_c):.3f})")
plt.legend()
plt.title("PR - Graph MultiHead")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/PR.png")
plt.close()

# =====================================================
# SAVE CONFIDENCE
# =====================================================
results_df = pd.DataFrame({
    "true_label": val_labels_array,
    "predicted_label": val_preds_array,
    "confidence": np.max(val_probs_array, axis=1)
})

for i in range(5):
    results_df[f"class_{i}_prob"] = val_probs_array[:, i]

results_df.to_csv(f"{RESULTS_DIR}/val_predictions_with_confidence.csv", index=False)

# =====================================================
# ATTENTION VISUALIZATION
# =====================================================
print("Generating graph visualizations...")

all_attention = []
all_labels_for_attn = []

model.eval()
with torch.no_grad():
    for swin_feat, lesion_feat, labels_batch in val_loader:
        swin_feat = swin_feat.to(device)
        lesion_feat = lesion_feat.to(device)

        outputs, attn_weights = model(swin_feat, lesion_feat, return_attention=True)

        attn_weights = attn_weights.cpu().numpy()
        attn_weights = attn_weights.mean(axis=1)  # average heads

        all_attention.append(attn_weights)
        all_labels_for_attn.extend(labels_batch.numpy())

all_attention = np.vstack(all_attention)
all_labels_for_attn = np.array(all_labels_for_attn)

lesion_names = ["MA", "HE", "EX", "SE"]

for class_id in range(5):

    mask = (all_labels_for_attn == class_id)
    if np.sum(mask) == 0:
        continue

    avg_attention = all_attention[mask].mean(axis=0)

    # Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(avg_attention,
                xticklabels=lesion_names,
                yticklabels=lesion_names,
                annot=True,
                fmt=".2f",
                cmap="viridis")
    plt.title(f"Attention Heatmap - Class {class_id}")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graph_attention_heatmap_class_{class_id}.png")
    plt.close()

    # Network graph
    G = nx.DiGraph()
    for i in range(4):
        for j in range(4):
            if avg_attention[i,j] > 0.05:
                G.add_edge(lesion_names[i], lesion_names[j],
                           weight=avg_attention[i,j])

    plt.figure(figsize=(6,6))
    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight']*5 for u,v in edges]
    nx.draw(G, pos, with_labels=True,
            node_size=3000,
            node_color="lightblue",
            width=weights,
            arrows=True)
    plt.title(f"Graph Interaction - Class {class_id}")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/graph_network_visualization_class_{class_id}.png")
    plt.close()

print("Graph Multi-Head Fusion complete.")
