"""
predict.py
-----------
Given any image:
  1. Detect whether a car is present (confidence threshold).
  2. Classify brand + model using the trained CarCNN.
  3. Generate probability plots:
       - Brand probability  (e.g. "Toyota")
       - Model probability  (e.g. "Camry")

Usage:
    python predict.py path/to/image.jpg
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")           # works both headless and with display
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ──────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────
IMG_SIZE          = (224, 224)
CONF_THRESHOLD    = 0.40   # confident classification threshold
NO_CAR_THRESH     = 0.15   # max softmax below this → image not a known car
                            # (uniform over 10 classes ≈ 0.10, 0.15 gives a small margin)

MODEL_PATH        = "car_classifier.h5"
CLASS_INDEX_PATH  = "class_indices.pkl"

# ──────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────
def _safe_path(path: str) -> str:
    path = os.path.normpath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {os.path.abspath(path)}")
    return path


def _load_image_array(img_path: str, size=IMG_SIZE) -> np.ndarray:
    """Load image → float32 array in [0,1]."""
    img = load_img(img_path, target_size=size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def _parse_brand_model(class_name: str):
    """
    Parse 'Brand Model Body Year' → (brand, model).
    Works for our 10-class dataset (first token = brand, second = model).
    """
    parts = class_name.split()
    brand = parts[0] if len(parts) >= 1 else class_name
    model = parts[1] if len(parts) >= 2 else ""
    return brand, model


# ──────────────────────────────────────────
#  ASSET LOADER  (load once, reuse everywhere)
# ──────────────────────────────────────────
def load_assets():
    """
    Load the trained CarCNN and class index.
    Returns:
        model        – full CarCNN (classification)
        inv_map      – dict {index: class_name}
        class_names  – list of class names ordered by index
    """
    model = load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "rb") as f:
        class_indices = pickle.load(f)

    inv_map     = {v: k for k, v in class_indices.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]
    return model, inv_map, class_names


# ──────────────────────────────────────────
#  STEP 1 – CAR DETECTION  (trained CarCNN)
# ──────────────────────────────────────────
def detect_car(probs: np.ndarray, inv_map: dict):
    """
    Decide whether a car is present using the trained model's softmax output.

    Why this works:
      The CarCNN was trained exclusively on car images, so its softmax always
      sums to 1.0 regardless of input.  When a non-car image is given, no class
      dominates and all probabilities sit near the uniform baseline (1/10 = 0.10).
      We treat  max_prob < NO_CAR_THRESH  as "no recognisable car".

    Tiers:
      max_prob < 0.15              → no car / not a known car
      0.15 <= max_prob < 0.40      → car detected, model uncertain (→ fuzzy fallback)
      max_prob >= 0.40             → car detected, confident classification

    Returns:
        (is_car: bool, max_prob: float, top_class: str)
    """
    top_idx  = int(np.argmax(probs))
    max_prob = float(probs[top_idx])
    is_car   = max_prob >= NO_CAR_THRESH
    return is_car, max_prob, inv_map[top_idx]


# ──────────────────────────────────────────
#  STEP 2 – CLASSIFICATION
# ──────────────────────────────────────────
def classify(model, img_path: str) -> np.ndarray:
    """
    Run the pre-loaded CarCNN on img_path.
    Returns softmax probabilities (length = num_classes).
    """
    arr = _load_image_array(img_path)
    return model.predict(arr, verbose=0)[0]


# ──────────────────────────────────────────
#  STEP 3 – POSSIBILITY PLOTS
# ──────────────────────────────────────────
def _aggregate_probs(probs, class_names, key_fn):
    """
    Aggregate softmax probabilities by a grouping key.
    key_fn(class_name) → string group label
    Returns dict: { group_label: total_probability }
    """
    agg = {}
    for i, cls in enumerate(class_names):
        group = key_fn(cls)
        agg[group] = agg.get(group, 0.0) + float(probs[i])
    return agg


def _bar_plot(ax, data: dict, title: str, xlabel: str, color: str):
    """Draw a horizontal bar chart on the given axes."""
    labels = list(data.keys())
    values = list(data.values())

    bars = ax.barh(labels, values, color=color, edgecolor="white", height=0.55)
    ax.set_xlim(0, min(1.05, max(values) * 1.35 + 0.05))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%",
            va="center", ha="left", fontsize=9
        )


def generate_plots(img_path: str, probs, class_names,
                   top_k_brands: int = 5, top_k_models: int = 5,
                   save_path: str = None):
    """
    Build and save:
      - The input image
      - Top brand possibility bar chart
      - Top model possibility bar chart
    """
    # Aggregate by brand & model
    brand_probs = _aggregate_probs(
        probs, class_names,
        key_fn=lambda cls: _parse_brand_model(cls)[0]
    )
    model_probs = _aggregate_probs(
        probs, class_names,
        key_fn=lambda cls: _parse_brand_model(cls)[1]
    )

    top_brands = dict(sorted(brand_probs.items(), key=lambda x: -x[1])[:top_k_brands])
    top_models = dict(sorted(model_probs.items(), key=lambda x: -x[1])[:top_k_models])

    # Figure: image | brand bars | model bars
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("CarCNN – Prediction Analysis", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(1, 3, wspace=0.4)

    ax_img = fig.add_subplot(gs[0])
    pil_img = load_img(img_path)
    ax_img.imshow(pil_img)
    ax_img.axis("off")
    ax_img.set_title("Input Image", fontsize=11)

    ax_brand = fig.add_subplot(gs[1])
    _bar_plot(ax_brand, top_brands,
              title="Brand Probability",
              xlabel="Probability", color="#4E79A7")

    ax_model = fig.add_subplot(gs[2])
    _bar_plot(ax_model, top_models,
              title="Model Probability",
              xlabel="Probability", color="#F28E2B")

    plt.tight_layout()
    out = save_path or (os.path.splitext(img_path)[0] + "_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")
    return out


# ──────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────
def predict_and_plot(img_path: str):
    img_path = _safe_path(img_path)
    print(f"\n{'─'*50}")
    print(f" Image : {os.path.abspath(img_path)}")
    print(f"{'─'*50}")

    # Load model and assets once
    print("\nLoading CarCNN and assets …")
    model, inv_map, class_names = load_assets()

    # Steps 1+2: run classifier, then apply detection threshold
    print("\n[1/3] Running CarCNN classifier …")
    probs = classify(model, img_path)

    print("\n[2/3] Checking car presence …")
    is_car, top_conf, top_cls = detect_car(probs, inv_map)
    brand, model_name = _parse_brand_model(top_cls)

    if not is_car:
        print(f"  ✘ No car detected — max confidence only {top_conf*100:.1f}%")
        print(f"    (Best guess was '{top_cls}' but below NO_CAR_THRESH={NO_CAR_THRESH*100:.0f}%)")
        _show_no_car_plot(img_path)
        return

    print(f"  ✔ Car detected  (max confidence: {top_conf*100:.1f}%)")

    if top_conf >= CONF_THRESHOLD:
        print(f"  ✔ Prediction  : {top_cls}")
        print(f"      Brand     : {brand}")
        print(f"      Model     : {model_name}")
        print(f"      Confidence: {top_conf*100:.1f}%")
    else:
        print(f"  ⚠ Uncertain ({top_conf*100:.1f}%) – top softmax predictions:")
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            print(f"    {inv_map[idx]:45s}  {probs[idx]*100:.1f}%")

    # Step 3: probability plots
    print("\n[3/3] Generating probability plots …")
    generate_plots(img_path, probs, class_names)

    print("\nDone.")


def _show_no_car_plot(img_path: str):
    """Display the image with a 'No Car Detected' overlay and save it."""
    fig, ax = plt.subplots(figsize=(6, 6))
    pil_img = load_img(img_path)
    ax.imshow(pil_img)
    ax.axis("off")
    ax.set_title("⚠  No Car Detected", fontsize=14, color="red", fontweight="bold")
    out = os.path.splitext(img_path)[0] + "_no_car.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved → {out}")


# ──────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    predict_and_plot(sys.argv[1])