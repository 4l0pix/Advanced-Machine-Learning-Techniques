"""
KOUKOSIAS ATHANASIOS-UTH-2026
DATASET:https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset
data have beeen heavily edited and augmented to fit the needs of the project:)
"""

import os
import pickle
import itertools
import random
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")          #headless-safe backend
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ----------------|CONFIG|---------------- 
DATA_DIR   = "DATA"  # directory containing brand sub-folders
IMG_SIZE   = (224, 224)
BATCH_SIZE = 64
EPOCHS     = 1
BASE_LR    = 0.001

MODEL_PATH       = "car_classifier.h5"
CLASS_INDEX_PATH = "class_indices.pkl"
METRICS_PLOT     = "training_metrics.png"
AUG_EXTRA        = 50      #target = min_class_count + AUG_EXTRA
AUG_PREFIX       = "aug_" #prefix for generated files so they can be removed later
# ---------------------------------------

"""
# --------|GPU SETUP|--------
def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[GPU] No GPU found – running on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"[GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")

    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[GPU] Mixed precision enabled (mixed_float16).")
    except Exception as e:
        print(f"[GPU] Mixed precision unavailable: {e}")
"""



# --------|AUGMENTATION|--------
def augment_to_balance(directory):
    """Generate augmented images so every class reaches max_count + AUG_EXTRA."""
    aug = ImageDataGenerator(
        rotation_range=30, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, zoom_range=0.2, horizontal_flip=True,
        brightness_range=(0.8, 1.2), fill_mode="nearest",
    )
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    counts  = {c: len(os.listdir(os.path.join(directory, c))) for c in classes}
    target  = max(counts.values()) + AUG_EXTRA

    for cls, cnt in counts.items():
        need = target - cnt
        if need <= 0:
            continue
        cls_dir = os.path.join(directory, cls)
        imgs = [f for f in os.listdir(cls_dir) if not f.startswith(AUG_PREFIX)]
        for i in range(need):
            src = os.path.join(cls_dir, imgs[i % len(imgs)])
            img = img_to_array(load_img(src, target_size=IMG_SIZE))
            img = aug.random_transform(img)
            out = os.path.join(cls_dir, f"{AUG_PREFIX}{i}_{imgs[i % len(imgs)]}")
            array_to_img(img).save(out)
        print(f"  {cls}: {cnt} → {target}  (+{need} augmented)")


# --------|TRAIN / TEST SPLIT|--------
def split_data(src_dir, ratio=0.8):
    """Split brand sub-folders into train/test sets inside src_dir by the given ratio."""
    train_dir = os.path.join(src_dir, "train")
    val_dir   = os.path.join(src_dir, "test")
    for cls in os.listdir(src_dir):
        cls_path = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_path) or cls in ("train", "test"):
            continue
        imgs = os.listdir(cls_path)
        random.shuffle(imgs)
        split = int(len(imgs) * ratio)
        for dest, subset in [(train_dir, imgs[:split]), (val_dir, imgs[split:])]:
            dest_cls = os.path.join(dest, cls)
            os.makedirs(dest_cls, exist_ok=True)
            for f in subset:
                shutil.copy2(os.path.join(cls_path, f), os.path.join(dest_cls, f))
        print(f"  {cls}: {split} train / {len(imgs) - split} test")


# --------|DATA GENERATORS|-------- 
def create_generators(train_dir, val_dir):
    #rescaling and augmentation are handled inside the model
    train_gen = ImageDataGenerator().flow_from_directory(
        train_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
    )
    val_gen = ImageDataGenerator().flow_from_directory(
        val_dir, target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    return train_gen, val_gen


# --------|MODEL|--------
def build_model(num_classes):
    """
    Sequential CNN – each block is structurally distinct:
      Block 1 (32)  – 3×3 convs + light Dropout
      Block 2 (64)  – 3×3 convs + 1×1 channel-mix conv
      Block 3 (128) – 5×5 wide-receptive conv + SpatialDropout2D
      Block 4 (256) – 3 convs + GlobalAveragePooling (no MaxPool)
      Head          – Dense(512, features) --> Dropout --> Softmax
    """
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(*IMG_SIZE, 3)),

        #augmentation–active only during training
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),

        # -----|Block 1(32 filters)–standardd 3×3,light dropout|-----
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),                              #spatial channel dropout

        # -----|Block 2(64 filters)–adds 1×1 channel-mix conv|-----
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 1, padding="same", activation="relu"),  # 1×1 squeeze
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # -----|Block 3(128 filters)–wider 5×5 receptive field|-----
        layers.Conv2D(128, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.SpatialDropout2D(0.2),                    #drops entire feature maps

        # -----|Block 4(256 filters)–3 convs, ends with GAP not MaxPool|-----
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),                 #averages spatial dims-->256-d vector

        # -----|Head|-----
        layers.Dense(512, activation="relu", name="features"),  #<-- feature layer for DB
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", name="softmax"),
    ], name="CarCNN")
    return model


# --------|TRAINING|-------- #
def train_model(model, train_gen, val_gen):
    #balanced class weights – compensates for uneven images-per-category
    cw = compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
    class_weight = dict(enumerate(cw))
    callbacks = [ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",save_best_only=True, verbose=1),EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]

    def _compile(lr):
        model.compile(
            optimizer=Adam(lr, clipnorm=1.0),
            loss=CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )

    print("\n---|Phase 1: Full CNN training|---")
    _compile(BASE_LR)
    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS // 2, callbacks=callbacks,
                   class_weight=class_weight)

    print("\n---|Phase 2: Fine-tuning (lower LR)|---")
    _compile(BASE_LR / 10)
    h2 = model.fit(train_gen, validation_data=val_gen,
                   epochs=EPOCHS, callbacks=callbacks,
                   class_weight=class_weight)

    return model, h1, h2


# --------|METRICS PLOT|-------- #
def save_metrics_plot(h1, h2, save_path=METRICS_PLOT):
    #merge both phase histories and plot accuracy + loss.
    def _merge(key):
        a = h1.history.get(key, [])
        b = h2.history.get(key, [])
        return a + b

    acc      = _merge("accuracy")
    val_acc  = _merge("val_accuracy")
    loss     = _merge("loss")
    val_loss = _merge("val_loss")
    epochs   = range(1, len(acc) + 1)
    split    = len(h1.history["accuracy"])  #phase boundary

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Metrics – CarCNN", fontsize=14, fontweight="bold")

    # --|Accuracy|--
    ax = axes[0]
    ax.plot(epochs, acc,     "b-o",  markersize=4, label="Train Acc")
    ax.plot(epochs, val_acc, "r-o",  markersize=4, label="Val Acc")
    ax.axvline(split, color="grey", linestyle="--", linewidth=1, label="Phase 2 start")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --|Loss|--
    ax = axes[1]
    ax.plot(epochs, loss,     "b-o",  markersize=4, label="Train Loss")
    ax.plot(epochs, val_loss, "r-o",  markersize=4, label="Val Loss")
    ax.axvline(split, color="grey", linestyle="--", linewidth=1, label="Phase 2 start")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Metrics plot saved --> {save_path}")


# =================|MAIN|=================
if __name__ == "__main__":
    #configure_gpu()        #uncomment if GPU setup is needed (optional) I dont use it cause im running VS-studio on GPU anyways

    print("=== Splitting dataset into train/test ===")
    split_data(DATA_DIR)
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR   = os.path.join(DATA_DIR, "test")

    print("=== Balancing class image counts ===")
    augment_to_balance(TRAIN_DIR)
    augment_to_balance(VAL_DIR)

    train_gen, val_gen = create_generators(TRAIN_DIR, VAL_DIR)

    with open(CLASS_INDEX_PATH, "wb") as f:
        pickle.dump(train_gen.class_indices, f)
    print(f"Saved class indices ({train_gen.num_classes} classes).")

    model = build_model(train_gen.num_classes)
    model.summary()

    model, h1, h2 = train_model(model, train_gen, val_gen)

    #save metrics plot
    save_metrics_plot(h1, h2)

    #sinal evaluation
    print("\n===|FINAL METRICS|===")
    preds  = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    print(classification_report(
        val_gen.classes,
        y_pred,
        target_names=list(train_gen.class_indices.keys())
    ))

