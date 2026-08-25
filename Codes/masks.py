import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp

# =========================
# CONFIG (EDIT IF NEEDED)
# =========================
DATA_ROOT = r"E:\NIT Delhi\Research 2\DDRseg"          # expects: {split}/image/*
SPLIT = "test"
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LESIONS = ["MA", "HE", "EX", "SE"]
NUM_CLASSES = 1 + len(LESIONS)

CHECKPOINT_PATH = r"best_lesion_segmenter_ddr.pth"    # can be relative to this script
OUT_DIR = r"E:\NIT Delhi\Research 2\pred_test_masks"   # output folder


# =========================
# PREPROCESSING (same as training)
# =========================
def ben_graham_preprocess(img):
    img = cv2.GaussianBlur(img, (0, 0), 10)
    return cv2.addWeighted(img, 4, img, -4, 128)

def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    img = img[y:y + h, x:x + w]

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = ben_graham_preprocess(img)

    img = (img / 255.0).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))  # (C,H,W)
    return torch.tensor(img, dtype=torch.float32)


# =========================
# MODEL (same as training)
# =========================
class DRLesionSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
            decoder_attention_type="scse",
        )

    def forward(self, x):
        return self.model(x)


# =========================
# INFERENCE
# =========================
@torch.no_grad()
def generate_and_save_masks(
    checkpoint_path=CHECKPOINT_PATH,
    data_root=DATA_ROOT,
    split=SPLIT,
    out_dir=OUT_DIR,
    save_color_index_mask=True,
):
    image_dir = os.path.join(data_root, split, "image")
    image_paths = sorted(glob(os.path.join(image_dir, "*")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    os.makedirs(out_dir, exist_ok=True)
    for lesion in LESIONS:
        os.makedirs(os.path.join(out_dir, lesion), exist_ok=True)
    if save_color_index_mask:
        os.makedirs(os.path.join(out_dir, "class_index"), exist_ok=True)

    model = DRLesionSegmenter().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    for img_path in tqdm(image_paths, desc="Generating binary masks"):
        name = os.path.basename(img_path)

        x = preprocess_image(img_path).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
        logits = model(x)                                       # (1,C,H,W)
        probs = torch.softmax(logits, dim=1)[0]                 # (C,H,W)

        pred_class = torch.argmax(probs, dim=0).byte().cpu().numpy()  # (H,W), 0..C-1

        # Save binary mask per lesion (mutually exclusive, derived from argmax)
        for i, lesion in enumerate(LESIONS, start=1):
            bin_mask = (pred_class == i).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out_dir, lesion, name), bin_mask)

        # Optional: save class-index mask (0..NUM_CLASSES-1) as PNG
        if save_color_index_mask:
            cv2.imwrite(os.path.join(out_dir, "class_index", name), pred_class)

    print(f"Done. Saved masks to: {out_dir}")


def main():
    generate_and_save_masks()


if __name__ == "__main__":
    main()