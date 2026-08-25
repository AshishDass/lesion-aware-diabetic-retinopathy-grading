import os
import cv2
import torch
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LESIONS = ["MA", "HE", "EX", "SE"]

# ---------------------------
# STEP 1: MODEL LOADING
# ---------------------------
# def load_segmentation_model(model_path):
#     model = torch.load(model_path, map_location=DEVICE)
#     model.eval()
#     return model

def _safe_torch_load(path, device):
    # Works across torch versions (and avoids the future weights_only warning when supported)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def _infer_num_classes_and_attention(state_dict, default_classes=5):
    # Try to infer output channels from SMP segmentation head weights
    for k in ("segmentation_head.0.weight", "segmentation_head.weight"):
        if k in state_dict:
            out_ch = int(state_dict[k].shape[0])
            break
    else:
        out_ch = default_classes

    # Heuristic: scSE attention layers add keys containing attention/scse
    uses_scse = any(("scse" in key.lower() or "attention" in key.lower()) for key in state_dict.keys())
    return out_ch, uses_scse

def load_segmentation_model(model_path):
    ckpt = _safe_torch_load(model_path, DEVICE)

    # support both formats:
    # 1) raw state_dict saved via torch.save(model.state_dict(), ...)
    # 2) dict checkpoint saved via torch.save({"model_state": ...}, ...)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state_dict, dict):
        # fallback: if someone saved the full model object
        state_dict.eval()
        return state_dict

    num_classes, use_scse = _infer_num_classes_and_attention(state_dict, default_classes=len(LESIONS) + 1)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # avoid trying to download ImageNet weights
        in_channels=3,
        classes=num_classes,
        activation=None,
        decoder_attention_type="scse" if use_scse else None,
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Warning: state_dict mismatch:")
        if missing:
            print("  Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  Unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.eval()
    return model

# ---------------------------
# STEP 1: INFERENCE
# ---------------------------
def infer_lesion_masks(model, image):
    """
    Returns dict of binary masks per lesion type
    """
    img = cv2.resize(image, (512, 512))
    img = img / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = model(img)  # shape: [1, 4, H, W]
        pred = torch.sigmoid(pred)[0].cpu().numpy()

    masks = {}
    for i, lesion in enumerate(LESIONS):
        masks[lesion] = (pred[i] > 0.5).astype(np.uint8)

    return masks


# ---------------------------
# RETINAL MASK
# ---------------------------
def get_retinal_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray > 10).astype(np.uint8)


# ---------------------------
# GEOMETRIC FEATURES
# ---------------------------
def lesion_burden_features(mask, retina_mask):
    labeled = label(mask)
    regions = regionprops(labeled)

    count = len(regions)
    total_area = mask.sum()
    retina_area = retina_mask.sum()

    avg_size = np.mean([r.area for r in regions]) if regions else 0
    area_ratio = total_area / (retina_area + 1e-6)

    return count, avg_size, area_ratio, regions


# ---------------------------
# SPATIAL FEATURES
# ---------------------------
def spatial_features(regions, macula_center, img_shape):
    h, w = img_shape
    fundus_radius = min(h, w) / 2

    distances = []
    quadrant_hits = [0, 0, 0, 0]

    for r in regions:
        y, x = r.centroid
        d = np.linalg.norm([x - macula_center[0], y - macula_center[1]])
        distances.append(d / fundus_radius)

        if x < macula_center[0] and y < macula_center[1]:
            quadrant_hits[0] += 1
        elif x >= macula_center[0] and y < macula_center[1]:
            quadrant_hits[1] += 1
        elif x < macula_center[0] and y >= macula_center[1]:
            quadrant_hits[2] += 1
        else:
            quadrant_hits[3] += 1

    return {
        "mean_dist": np.mean(distances) if distances else 0,
        "min_dist": np.min(distances) if distances else 1,
        "quad_count": sum(q > 0 for q in quadrant_hits)
    }


# ---------------------------
# CENTRAL vs PERIPHERAL
# ---------------------------
def central_ratio(mask, macula_center, img_shape):
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - macula_center[0])**2 + (Y - macula_center[1])**2)
    central_zone = dist < 0.3 * min(h, w)

    central = np.logical_and(mask, central_zone).sum()
    total = mask.sum()

    return central / (total + 1e-6)


# ---------------------------
# ICDR RULE FLAGS
# ---------------------------
def icdr_flags(features):
    return {
        "flag_mild_DR": int(features["count_MA"] > 0 and features["area_ratio_HE"] == 0),
        "flag_severe_NPDR": int(features["quad_HE"] >= 2 or features["count_SE"] > 5),
        "flag_vision_threat": int(features["min_dist_EX"] < 0.2 and features["avg_size_EX"] > 50)
    }


# ---------------------------
# MASTER EXTRACTION
# ---------------------------
def extract_features(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))

    retina_mask = get_retinal_mask(image)
    macula_center = (256, 256)

    lesion_masks = infer_lesion_masks(model, image)

    features = {}

    for lesion in LESIONS:
        count, avg_size, area_ratio, regions = lesion_burden_features(
            lesion_masks[lesion], retina_mask
        )

        spatial = spatial_features(regions, macula_center, image.shape[:2])
        central = central_ratio(lesion_masks[lesion], macula_center, image.shape[:2])

        features.update({
            f"count_{lesion}": count,
            f"avg_size_{lesion}": avg_size,
            f"area_ratio_{lesion}": area_ratio,
            f"mean_dist_{lesion}": spatial["mean_dist"],
            f"min_dist_{lesion}": spatial["min_dist"],
            f"quad_{lesion}": spatial["quad_count"],
            f"central_ratio_{lesion}": central
        })

    features.update(icdr_flags(features))
    features["image_name"] = os.path.basename(image_path)

    return features


# ---------------------------
# BATCH PROCESS
# ---------------------------
def run_step2(image_dir, model_path, output_csv):
    model = load_segmentation_model(model_path)
    all_features = []

    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(image_dir, img_name)
            feats = extract_features(img_path, model)
            all_features.append(feats)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    run_step2(
        image_dir="sample_images/",
        # model_path="step1_model/lesion_model.pth",
        model_path="best_model_final.pth",
        output_csv="output/lesion_features.csv"
    )
