# CarCNN – Code Documentation
**KOUKOSIAS ATHANASIOS – UTH 2026**

---

## Overview

`neural-network.py` implements a full pipeline for car brand classification using a custom Sequential CNN built with TensorFlow/Keras. The pipeline covers dataset splitting, class balancing via augmentation, model construction, two-phase training, and evaluation.

---

## Configuration Constants

| Constant | Default | Description |
|---|---|---|
| `DATA_DIR` | `"DATA"` | Root directory containing one sub-folder per car brand |
| `IMG_SIZE` | `(224, 224)` | All images are resized to this resolution |
| `BATCH_SIZE` | `64` | Number of images per training batch |
| `EPOCHS` | `50` | Total epochs across both training phases |
| `BASE_LR` | `0.0001` | Learning rate for Phase 1; Phase 2 uses `BASE_LR / 10` |
| `MODEL_PATH` | `"car_classifier.h5"` | Where the best model checkpoint is saved |
| `CLASS_INDEX_PATH` | `"class_indices.pkl"` | Pickle file mapping class names → integer indices |
| `METRICS_PLOT` | `"training_metrics.png"` | Output path for the training metrics plot |
| `AUG_EXTRA` | `50` | Extra images added on top of the largest class count |
| `AUG_PREFIX` | `"aug_"` | Filename prefix for generated augmentation files |

---

## Functions

### `augment_to_balance(directory)`

**Purpose:** Equalises image counts across all classes in a directory by generating synthetic augmented images for under-represented classes.

**Parameters:**
- `directory` *(str)* – Path to a folder whose direct sub-folders are class directories (e.g. `DATA/train/`).

**How it works:**
1. Lists all sub-folders (classes) and counts their images.
2. Sets a `target = max_count + AUG_EXTRA`.
3. For each class that has fewer images than `target`, it repeatedly picks source images (cycling if needed) and applies random transforms:
   - Rotation up to ±30°
   - Width/height shifts up to 15%
   - Shear up to 15%
   - Zoom up to 20%
   - Horizontal flip
   - Brightness variation ×0.8–1.2
4. Saves each generated image with the `AUG_PREFIX` prefix so it can be identified and cleaned up later.

**Side effects:** Writes new `.jpg` files into the class sub-folders of `directory`.

---

### `split_data(src_dir, ratio=0.8)`

**Purpose:** Performs a stratified train/test split on a flat branded dataset, producing `train/` and `test/` sub-folders inside `src_dir`.

**Parameters:**
- `src_dir` *(str)* – Root directory containing one sub-folder per brand (e.g. `DATA/`).
- `ratio` *(float)* – Fraction of images assigned to training. Default `0.8` (80% train / 20% test).

**How it works:**
1. Derives `train_dir = src_dir/train` and `val_dir = src_dir/test`.
2. Iterates every sub-folder that is not already named `train` or `test`.
3. Shuffles the image list randomly.
4. Copies the first `ratio × N` images to `train/<class>/` and the remainder to `test/<class>/`.

**Notes:**
- Original images are **copied**, not moved — the source brand folders remain intact.
- Calling this twice on the same directory is safe because `os.makedirs(..., exist_ok=True)` is used and files will simply be overwritten.

---

### `create_generators(train_dir, val_dir)`

**Purpose:** Creates Keras `ImageDataGenerator` flow objects for the training and validation sets.

**Parameters:**
- `train_dir` *(str)* – Path to the training split (e.g. `DATA/train/`).
- `val_dir` *(str)* – Path to the validation/test split (e.g. `DATA/test/`).

**Returns:** `(train_gen, val_gen)` — two `DirectoryIterator` objects.

**How it works:**
- No pixel-level preprocessing is applied here (`ImageDataGenerator()` with no arguments) because `Rescaling(1/255)` and random augmentation layers are baked into the model itself.
- `train_gen` uses `shuffle=True`; `val_gen` uses `shuffle=False` to keep evaluation order deterministic.
- Both generators resize images to `IMG_SIZE` and use `class_mode="categorical"` (one-hot labels).

---

### `build_model(num_classes)`

**Purpose:** Constructs and returns the `CarCNN` Sequential model.

**Parameters:**
- `num_classes` *(int)* – Number of output classes (brands), inferred from the training generator.

**Returns:** A compiled-ready `keras.Sequential` model named `CarCNN`.

**Architecture:**

```
Input (224×224×3)
│
├── Rescaling(1/255)               ← normalise pixels to [0,1]
│
├── RandomFlip / RandomRotation    ← augmentation (training only)
│   RandomZoom / RandomContrast
│
├── Block 1 – 32 filters
│   Conv2D(32,3) → BN → Conv2D(32,3) → BN → MaxPool → Dropout(0.1)
│
├── Block 2 – 64 filters
│   Conv2D(64,3) → BN → Conv2D(64,3) → BN → Conv2D(64,1) → BN → MaxPool
│   (1×1 conv acts as a channel-mixing / squeeze layer)
│
├── Block 3 – 128 filters
│   Conv2D(128,5) → BN → Conv2D(128,3) → BN → MaxPool → SpatialDropout2D(0.2)
│   (5×5 kernel captures wider spatial context)
│
├── Block 4 – 256 filters
│   Conv2D(256,3) → BN → Conv2D(256,3) → BN → Conv2D(256,3) → BN
│   → GlobalAveragePooling2D   ← replaces Flatten, reduces parameters
│
└── Head
    Dense(512, relu, name="features")   ← feature vector for retrieval/DB use
    → Dropout(0.5)
    → Dense(num_classes, softmax)
```

**Design choices:**
- `GlobalAveragePooling2D` instead of `Flatten` cuts the parameter count in the head and improves spatial invariance.
- In-model augmentation layers (`RandomFlip` etc.) are automatically disabled during `model.predict()` and `model.evaluate()`.
- The `"features"` layer can be used as an embedding extractor for similarity search.

---

### `train_model(model, train_gen, val_gen)`

**Purpose:** Trains the model in two phases with decreasing learning rate, using class-balanced weights and callbacks.

**Parameters:**
- `model` – The Keras model returned by `build_model`.
- `train_gen` – Training data generator.
- `val_gen` – Validation data generator.

**Returns:** `(model, h1, h2)` — the trained model and the two `History` objects from each phase.

**How it works:**

| | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | `EPOCHS // 2` | `EPOCHS` |
| Learning rate | `BASE_LR` | `BASE_LR / 10` |
| Purpose | Train all layers from scratch | Fine-tune with smaller updates |

**Callbacks:**
- `ModelCheckpoint` — saves the model to `MODEL_PATH` only when `val_accuracy` improves.
- `EarlyStopping` — monitors `val_loss` and stops after 10 epochs without improvement, restoring the best weights.

**Class weights:** `compute_class_weight("balanced", ...)` assigns higher loss weight to under-represented brands, preventing the model from biasing toward large classes.

**Optimiser:** Adam with `clipnorm=1.0` (gradient clipping) prevents exploding gradients.

**Loss:** `CategoricalCrossentropy` with `label_smoothing=0.05` — softens hard targets slightly, reducing overconfidence and improving generalisation.

---

### `save_metrics_plot(h1, h2, save_path=METRICS_PLOT)`

**Purpose:** Merges the histories from both training phases and saves a two-panel plot (accuracy + loss) to disk.

**Parameters:**
- `h1` – `History` object from Phase 1.
- `h2` – `History` object from Phase 2.
- `save_path` *(str)* – Output file path. Defaults to `METRICS_PLOT`.

**How it works:**
1. Concatenates Phase 1 and Phase 2 values for `accuracy`, `val_accuracy`, `loss`, `val_loss`.
2. Draws a dashed vertical line at the epoch where Phase 2 begins.
3. Saves the figure at 150 DPI and closes it (headless-safe via the `Agg` backend).

---

## Main Pipeline (`__main__`)

```
1. split_data(DATA_DIR)
       │
       └── Creates DATA/train/<brand>/ and DATA/test/<brand>/

2. augment_to_balance(TRAIN_DIR)
   augment_to_balance(VAL_DIR)
       │
       └── Equalises class counts in both splits

3. create_generators(TRAIN_DIR, VAL_DIR)
       │
       └── Keras iterators ready for training

4. pickle.dump(class_indices)
       └── Saves class-name → index mapping for inference scripts

5. build_model(num_classes)
   model.summary()

6. train_model(model, train_gen, val_gen)
       │
       ├── Phase 1 (LR = BASE_LR)
       └── Phase 2 (LR = BASE_LR / 10)

7. save_metrics_plot(h1, h2)

8. classification_report(val_gen, predictions)
```
