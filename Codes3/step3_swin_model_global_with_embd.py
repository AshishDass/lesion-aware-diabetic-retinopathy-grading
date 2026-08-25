import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_curve, precision_recall_curve
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMAGE_DIR = "/Users/adass/Research/aptos2019-blindness-detection/train_images"
CSV_PATH = "/Users/adass/Research/aptos2019-blindness-detection/train.csv"

BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_CLASSES = 5
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4
IMAGE_SIZE = 384
EARLY_STOPPING_PATIENCE = 5

RESULTS_DIR = "results_global_only"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DATASET CLASS
# ============================================================

class APTOSDataset(Dataset):
    """
    Custom dataset for loading APTOS images.
    Reads image file using id_code and assigns diagnosis label.
    """

    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["id_code"] + ".png")
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Missing image: {img_path}")
            return None, None
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None
        label = row["diagnosis"]

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Mild augmentation suitable for retinal images
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# Validation transform without augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ============================================================
# MODEL DEFINITION
# ============================================================

def get_model():
    """
    Loads pretrained Swin-Tiny model.
    Freezes early transformer stages to reduce overfitting.
    Replaces classification head for 5-class DR grading.
    """
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=NUM_CLASSES,
        img_size=384
    )

    # Freeze early feature extraction layers
    for name, param in model.named_parameters():
        if "layers.0" in name or "layers.1" in name:
            param.requires_grad = False

    return model

# ============================================================
# METRIC COMPUTATION FUNCTION
# ============================================================

def compute_metrics(labels, preds, probs):
    """
    Computes all required classification metrics.
    """
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
        "f1": f1_score(labels, preds, average="macro"),
        "auc": roc_auc_score(labels, probs, multi_class="ovr"),
        "mcc": matthews_corrcoef(labels, preds),
        "kappa": cohen_kappa_score(labels, preds)
    }

# ============================================================
# TRAINING FUNCTION
# ============================================================

def run_training(resume_path=None):

    df = pd.read_csv(CSV_PATH)

    # Stratified 80-20 split to preserve class balance
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["diagnosis"],
        random_state=42
    )

    train_dataset = APTOSDataset(train_df, IMAGE_DIR, train_transform)
    val_dataset = APTOSDataset(val_df, IMAGE_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Separate backbone and head parameters for differential learning rate
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD}
    ], weight_decay=0.01)


    # ---- LOAD CHECKPOINT IF RESUME ----
    best_f1 = 0
    early_stop_counter = 0
    epoch_metrics = []
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE), strict=False)

    for epoch in range(NUM_EPOCHS):

        # ================= TRAINING =================
        model.train()
        train_preds, train_labels, train_probs = [], [], []

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(probs.detach().cpu().numpy())

        train_metrics = compute_metrics(train_labels, train_preds, train_probs)

        # ================= VALIDATION =================
        model.eval()
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_metrics = compute_metrics(val_labels, val_preds, val_probs)

        print(f"\nEpoch {epoch+1}")
        print(f"Train F1: {train_metrics['f1']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")

        # Save metrics per epoch
        epoch_metrics.append({
            "epoch": epoch+1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        # Early stopping based on validation F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            early_stop_counter = 0
            torch.save(model.state_dict(),
                       f"{RESULTS_DIR}/best_model.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Save epoch-wise metrics
    pd.DataFrame(epoch_metrics).to_csv(
        f"{RESULTS_DIR}/epoch_metrics.csv", index=False
    )

    print("Training complete.")
    return val_labels, val_preds, val_probs, val_df

# ============================================================
# REPORT GENERATION (ROC, PR, CONFIDENCE)
# ============================================================

# def generate_reports(val_labels, val_preds, val_probs, val_df):

#     val_labels = np.array(val_labels)
#     val_preds = np.array(val_preds)
#     val_probs = np.array(val_probs)

#     # ============================================================
#     # FINAL METRIC COMPUTATION (Best Model Evaluation)
#     # ============================================================

#     final_metrics = compute_metrics(val_labels, val_preds, val_probs)

#     print("\n========== FINAL PUBLISHABLE RESULTS ==========")
#     for k, v in final_metrics.items():
#         print(f"{k.upper()}: {v:.4f}")

#     # Save final results in TXT format (paper friendly)
#     with open(f"{RESULTS_DIR}/final_publishable_results.txt", "w") as f:
#         f.write("FINAL GLOBAL ONLY MODEL RESULTS\n")
#         f.write("=================================\n")
#         for k, v in final_metrics.items():
#             f.write(f"{k.upper()}: {v:.6f}\n")

#     # Save final results in CSV format
#     pd.DataFrame([final_metrics]).to_csv(
#         f"{RESULTS_DIR}/final_publishable_results.csv",
#         index=False
#     )

#     # ============================================================
#     # ROC CURVE
#     # ============================================================

#     plt.figure()
#     for i in range(NUM_CLASSES):
#         fpr, tpr, _ = roc_curve(val_labels == i, val_probs[:, i])
#         plt.plot(fpr, tpr, label=f"Class {i}")
#     plt.plot([0,1],[0,1],'--')
#     plt.legend()
#     plt.title("ROC Curve - Global Only")
#     plt.savefig(f"{RESULTS_DIR}/ROC_curve.png")
#     plt.close()

#     # ============================================================
#     # PR CURVE
#     # ============================================================

#     plt.figure()
#     for i in range(NUM_CLASSES):
#         precision_c, recall_c, _ = precision_recall_curve(
#             val_labels == i,
#             val_probs[:, i]
#         )
#         plt.plot(recall_c, precision_c, label=f"Class {i}")
#     plt.legend()
#     plt.title("PR Curve - Global Only")
#     plt.savefig(f"{RESULTS_DIR}/PR_curve.png")
#     plt.close()

#     # ============================================================
#     # SAVE CONFIDENCE SCORES
#     # ============================================================

#     results_df = val_df.copy()
#     results_df["predicted_label"] = val_preds
#     results_df["confidence"] = np.max(val_probs, axis=1)

#     for i in range(NUM_CLASSES):
#         results_df[f"class_{i}_prob"] = val_probs[:, i]

#     results_df.to_csv(
#         f"{RESULTS_DIR}/val_predictions_with_confidence.csv",
#         index=False
#     )

#     print("Reports and final publishable results generated.")


def generate_reports(val_labels, val_preds, val_probs, val_df):

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import auc

    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    val_probs = np.array(val_probs)

    # ================= FINAL METRICS =================
    final_metrics = compute_metrics(val_labels, val_preds, val_probs)

    print("\n========== FINAL PUBLISHABLE RESULTS ==========")
    for k, v in final_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    with open(f"{RESULTS_DIR}/final_publishable_results.txt", "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k.upper()}: {v:.6f}\n")

    pd.DataFrame([final_metrics]).to_csv(
        f"{RESULTS_DIR}/final_publishable_results.csv",
        index=False
    )

    # ================= ROC CURVE =================
    val_labels_bin = label_binarize(val_labels, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8,6))

    # Per-class ROC
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(val_labels_bin[:, i], val_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.3f})")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(
        val_labels_bin.ravel(),
        val_probs.ravel()
    )
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro,
             linestyle='--',
             linewidth=2,
             label=f"Micro-average (AUC={roc_auc_micro:.3f})")

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([
        roc_curve(val_labels_bin[:, i], val_probs[:, i])[0]
        for i in range(NUM_CLASSES)
    ]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(val_labels_bin[:, i], val_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= NUM_CLASSES
    roc_auc_macro = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr,
             linestyle='-.',
             linewidth=2,
             label=f"Macro-average (AUC={roc_auc_macro:.3f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Global Only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ROC_curve.png")
    plt.close()

    # ================= PR CURVE =================
    plt.figure(figsize=(8,6))

    # Per-class PR
    for i in range(NUM_CLASSES):
        precision_c, recall_c, _ = precision_recall_curve(
            val_labels_bin[:, i],
            val_probs[:, i]
        )
        pr_auc = auc(recall_c, precision_c)
        plt.plot(recall_c, precision_c,
                 label=f"Class {i} (AUC={pr_auc:.3f})")

    # Micro-average PR
    precision_micro, recall_micro, _ = precision_recall_curve(
        val_labels_bin.ravel(),
        val_probs.ravel()
    )
    pr_auc_micro = auc(recall_micro, precision_micro)

    plt.plot(recall_micro, precision_micro,
             linestyle='--',
             linewidth=2,
             label=f"Micro-average (AUC={pr_auc_micro:.3f})")

    # Macro-average PR
    all_recall = np.linspace(0,1,100)
    mean_precision = np.zeros_like(all_recall)

    for i in range(NUM_CLASSES):
        precision_c, recall_c, _ = precision_recall_curve(
            val_labels_bin[:, i],
            val_probs[:, i]
        )
        mean_precision += np.interp(all_recall,
                                    recall_c[::-1],
                                    precision_c[::-1])

    mean_precision /= NUM_CLASSES
    pr_auc_macro = auc(all_recall, mean_precision)

    plt.plot(all_recall, mean_precision,
             linestyle='-.',
             linewidth=2,
             label=f"Macro-average (AUC={pr_auc_macro:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve - Global Only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/PR_curve.png")
    plt.close()

    # ================= CONFIDENCE =================
    results_df = val_df.copy()
    results_df["predicted_label"] = val_preds
    results_df["confidence"] = np.max(val_probs, axis=1)

    for i in range(NUM_CLASSES):
        results_df[f"class_{i}_prob"] = val_probs[:, i]

    results_df.to_csv(
        f"{RESULTS_DIR}/val_predictions_with_confidence.csv",
        index=False
    )

    print("Reports and publishable results generated.")



def extract_embeddings(model_path, output_csv):

    print("Extracting 768-d embeddings...")

    df = pd.read_csv(CSV_PATH)

    dataset = APTOSDataset(df, IMAGE_DIR, val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load backbone without classification head
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=0,
        img_size=384
    )

    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(DEVICE)
    model.eval()

    embeddings_list = []
    id_codes_list = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader)):
            valid_indices = [i for i, img in enumerate(images) if img is not None]
            if len(valid_indices) == 0:
                continue
            valid_images = torch.stack([images[i] for i in valid_indices]).to(DEVICE)
            valid_id_codes = [df["id_code"].iloc[batch_idx * BATCH_SIZE + i] for i in valid_indices]
            embeddings = model(valid_images)
            embeddings_list.append(embeddings.cpu().numpy())
            id_codes_list.extend(valid_id_codes)

    if len(embeddings_list) == 0:
        print("No embeddings extracted.")
        return

    embeddings = np.vstack(embeddings_list)
    feature_df = pd.DataFrame(embeddings)
    feature_df["id_code"] = id_codes_list

    feature_df.to_csv(output_csv, index=False)

    print(f"Embedding extraction complete. Total processed images: {len(id_codes_list)}")

 # ____________USAGE_____________

# if __name__ == "__main__":
#     val_labels, val_preds, val_probs, val_df = run_training()
#     generate_reports(val_labels, val_preds, val_probs, val_df)

#     # Extract embeddings from best model
#     extract_embeddings(
#         f"{RESULTS_DIR}/best_model.pth",
#         f"{RESULTS_DIR}/swin_embeddings.csv"
#     )



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # val_labels, val_preds, val_probs, val_df = run_training()
    # val_labels, val_preds, val_probs, val_df = run_training(resume_path=f"{RESULTS_DIR}_1/best_model.pth")
    # generate_reports(val_labels, val_preds, val_probs, val_df)
    extract_embeddings(
        f"{RESULTS_DIR}/best_model.pth",
        f"swin_embeddings.csv"
    )


######## without _1 is the final version 