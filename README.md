# Credit Card Fraud Detection — Autoencoder Documentation

**Dataset:** [Kaggle — Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Dataset](#2-dataset)
- [3. Core Concept: Anomaly Detection via Reconstruction Error](#3-core-concept-anomaly-detection-via-reconstruction-error)
- [4. Autoencoder Architecture](#4-autoencoder-architecture)
  - [4.1 Encoder](#41-encoder)
  - [4.2 Bottleneck](#42-bottleneck)
  - [4.3 Decoder](#43-decoder)
  - [4.4 Layer Summary](#44-layer-summary)
- [5. Function Reference](#5-function-reference)
  - [5.1 load_data](#51-load_data)
  - [5.2 preprocess](#52-preprocess)
  - [5.3 build_autoencoder](#53-build_autoencoder)
  - [5.4 train_model](#54-train_model)
  - [5.5 compute_reconstruction_error](#55-compute_reconstruction_error)
  - [5.6 find_best_threshold](#56-find_best_threshold)
  - [5.7 evaluate](#57-evaluate)
  - [5.8 plot_results](#58-plot_results)
  - [5.9 main](#59-main)
- [6. Results Interpretation](#6-results-interpretation)
  - [6.1 Final Validation Loss](#61-final-validation-loss)
  - [6.2 ROC-AUC Score](#62-roc-auc-score)
  - [6.3 Threshold](#63-threshold)
  - [6.4 Classification Report](#64-classification-report)
  - [6.5 Confusion Matrix](#65-confusion-matrix)
- [7. Tradeoffs and Limitations](#7-tradeoffs-and-limitations)

---

## 1. Overview

This project detects credit card fraud using an **autoencoder neural network** trained exclusively on legitimate (normal) transactions. Rather than learning to classify fraud directly, the model learns what a normal transaction looks like and flags anything it cannot reconstruct accurately as suspicious.

The pipeline follows these steps:

1. Load and normalize the dataset
2. Split into a training set (normal only) and a test set (normal + fraud)
3. Build and train a symmetric autoencoder
4. Compute reconstruction error for all test samples
5. Find the MSE threshold that best separates fraud from normal
6. Evaluate using precision, recall, F1-score, and ROC-AUC

---

## 2. Dataset

The dataset contains European credit card transactions from September 2013, provided by the Machine Learning Group at ULB.

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Normal transactions | 284,315 (99.83%) |
| Fraudulent transactions | 492 (0.17%) |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Label column | `Class` (0 = normal, 1 = fraud) |

Because the original features are sensitive, they have been transformed using PCA and are provided as anonymized components V1 through V28. `Time` and `Amount` are the only features with an original meaning. The `Class` column is the ground-truth label used only for evaluation — it is never seen during training.

The dataset is highly **imbalanced**: fraud accounts for only 0.17% of all transactions. This makes standard classifiers unreliable and is exactly why an anomaly-detection approach is well-suited here.

---

## 3. Core Concept: Anomaly Detection via Reconstruction Error

An autoencoder is trained to compress an input into a compact representation (the bottleneck) and then reconstruct the original input as accurately as possible. When trained only on normal transactions, it becomes very good at reconstructing normal patterns but poor at reconstructing patterns it has never seen — i.e., fraud.

The **reconstruction error** for a single sample is computed as the Mean Squared Error (MSE) between the original input and its reconstruction:

```
MSE = (1/n) * Σ (x_i - x̂_i)²
```

Where `x` is the original feature vector and `x̂` is the reconstructed vector.

- **Normal transactions** → low reconstruction error (the model knows how to reconstruct them)
- **Fraudulent transactions** → high reconstruction error (the model has never seen this pattern)

A **threshold** is chosen such that samples above it are classified as fraud. This threshold is not fixed — it is determined automatically by finding the value that maximises the F1-score on the test set.

---

## 4. Autoencoder Architecture

The autoencoder is a fully connected (dense) neural network with a symmetric structure. The input and output dimensions are both equal to the number of features (30 after preprocessing).

```
Input (30)
    │
    ▼
[Encoder Layer 1]  Dense(32, relu)
    │
    ▼
[Encoder Layer 2]  Dense(16, relu)
    │
    ▼
[Bottleneck]       Dense(8, relu)       ← compressed representation
    │
    ▼
[Decoder Layer 1]  Dense(16, relu)
    │
    ▼
[Decoder Layer 2]  Dense(32, relu)
    │
    ▼
Output (30)        Dense(30, linear)
```

### 4.1 Encoder

The encoder compresses the 30-dimensional input into progressively smaller representations: 32 → 16 → 8. Each layer uses the **ReLU** activation function, which introduces non-linearity while avoiding the vanishing gradient problem. ReLU outputs zero for negative values and passes positive values unchanged, allowing the network to learn which features are most relevant for capturing the structure of normal transactions.

### 4.2 Bottleneck

The bottleneck layer has 8 neurons — less than one third of the input size. This forced compression means the network cannot simply memorise inputs; it must learn a compact, generalisable representation of what makes a transaction normal. If a transaction cannot be compressed into this 8-dimensional space and then reconstructed, it is likely anomalous.

### 4.3 Decoder

The decoder mirrors the encoder in reverse — 8 → 16 → 32 — and ends with a **linear** output layer (no activation). The linear activation is appropriate here because the input features have been standardised and can take any real value. Using ReLU in the output would clip negative values and introduce systematic reconstruction error.

### 4.4 Layer Summary

| Layer | Type | Units | Activation | Parameters |
|---|---|---|---|---|
| input | Input | 30 | — | 0 |
| enc_32 | Dense | 32 | ReLU | 992 |
| enc_16 | Dense | 16 | ReLU | 528 |
| bottleneck | Dense | 8 | ReLU | 136 |
| dec_16 | Dense | 16 | ReLU | 144 |
| dec_32 | Dense | 32 | ReLU | 544 |
| output | Dense | 30 | Linear | 990 |

**Total trainable parameters: 3,334**

The model is intentionally small. Fraud detection on tabular data does not require a deep or wide network; keeping it compact reduces overfitting and makes training fast.

---

## 5. Function Reference

### 5.1 `load_data`

```python
def load_data(path):
```

**Purpose:** Reads the CSV file from the path provided by `kagglehub` and returns a pandas DataFrame.

**Parameters:**
- `path` — directory path returned by `kagglehub.dataset_download()`

**Returns:** `df` — the full dataset as a DataFrame

**Prints:** Total sample count, number of normal and fraudulent transactions, and the fraud rate as a percentage.

**Note:** No filtering or transformation happens here. This function is solely responsible for loading raw data and giving an initial statistical summary.

---

### 5.2 `preprocess`

```python
def preprocess(df):
```

**Purpose:** Scales all features using standard normalisation and creates the train/test split following the anomaly-detection paradigm.

**Parameters:**
- `df` — the raw DataFrame returned by `load_data`

**Returns:** `X_train`, `X_test`, `y_test`, `input_dim`

**Steps:**
1. Separates features (V1–V28, Time, Amount) from the label (`Class`)
2. Applies `StandardScaler` to normalize all features to zero mean and unit variance — this is important because the network uses MSE loss, which is sensitive to scale differences
3. Splits normal transactions 80/20 into train and validation-normal sets
4. Constructs `X_test` by concatenating the held-out normal transactions with **all** fraud transactions
5. Constructs `y_test` as the corresponding ground-truth labels (0 = normal, 1 = fraud)

**Why train only on normal transactions?** The autoencoder must learn the distribution of normal behaviour. If fraud examples were included in training, the model would also learn to reconstruct fraud accurately, destroying its ability to detect it through reconstruction error.

---

### 5.3 `build_autoencoder`

```python
def build_autoencoder(input_dim):
```

**Purpose:** Constructs and compiles the autoencoder model using the Keras functional API.

**Parameters:**
- `input_dim` — number of input features (30 in this dataset)

**Returns:** a compiled Keras `Model`

**Compilation settings:**
- **Optimizer:** Adam — an adaptive learning rate optimizer well-suited for sparse, noisy data
- **Loss:** Mean Squared Error (MSE) — directly measures reconstruction quality; minimising it during training teaches the model to reconstruct normal transactions as accurately as possible

The functional API is used (rather than `Sequential`) to keep the architecture explicit and easy to modify or extend.

---

### 5.4 `train_model`

```python
def train_model(model, X_train, epochs=30, batch_size=256):
```

**Purpose:** Trains the autoencoder using the training set, where both input and target are `X_train` (the model learns to reproduce its own input).

**Parameters:**
- `model` — the compiled autoencoder
- `X_train` — array of normal transaction feature vectors
- `epochs` — maximum number of training epochs (default: 30)
- `batch_size` — number of samples per gradient update (default: 256)

**Returns:** Keras `History` object containing loss values per epoch

**Key detail — `EarlyStopping`:** Training stops automatically if the validation loss does not improve for 5 consecutive epochs, and the best weights are restored. This prevents overfitting and makes the `epochs=30` parameter a ceiling rather than a fixed duration.

**`verbose=0`:** Training progress is suppressed to avoid clutter; only the final validation loss is printed after training completes.

---

### 5.5 `compute_reconstruction_error`

```python
def compute_reconstruction_error(model, X):
```

**Purpose:** Passes samples through the trained autoencoder and calculates the per-sample MSE between the original and reconstructed inputs.

**Parameters:**
- `model` — the trained autoencoder
- `X` — array of feature vectors (can be test set or any subset)

**Returns:** 1D array of reconstruction errors, one value per sample

**Formula:** For each sample, `error = mean((x - x_reconstructed)²)` across all 30 features.

This function is the core of the anomaly detection mechanism. The resulting array is used both for threshold selection and for ROC-AUC computation.

---

### 5.6 `find_best_threshold`

```python
def find_best_threshold(errors, y_true):
```

**Purpose:** Searches for the MSE threshold value that maximises the F1-score for the fraud class.

**Parameters:**
- `errors` — reconstruction error array from `compute_reconstruction_error`
- `y_true` — ground-truth labels for the test set

**Returns:** the best threshold as a float

**Method:** Candidate thresholds are generated at every 0.5th percentile between the 80th and 100th percentiles of the error distribution — a range where fraud samples are likely to be concentrated. For each candidate, precision, recall, and F1 for the fraud class are computed manually (avoiding sklearn's built-in to keep the logic transparent). The threshold with the highest F1 is returned.

**Why percentile-based?** Using absolute error values would make the threshold sensitive to the specific scale of the data. Percentile-based search is more robust and ensures the threshold is always in a meaningful range relative to the observed errors.

---

### 5.7 `evaluate`

```python
def evaluate(model, X_test, y_test):
```

**Purpose:** Runs the full evaluation pipeline: computes reconstruction errors, finds the best threshold, classifies samples, and prints all metrics.

**Parameters:**
- `model` — trained autoencoder
- `X_test` — test set feature vectors
- `y_test` — ground-truth labels

**Returns:** `errors`, `threshold`, `auc`, `y_pred`

**Metrics printed:**
- **ROC-AUC** — computed using raw reconstruction errors (threshold-independent)
- **Best threshold** — the MSE value above which a transaction is flagged as fraud
- **Classification report** — precision, recall, F1-score, and support for both classes
- **Confusion matrix** — broken down into TP, TN, FP, FN with plain-language labels

---

### 5.8 `plot_results`

```python
def plot_results(history, errors, y_test, threshold, auc):
```

**Purpose:** Generates and saves three diagnostic plots to `results.png`.

**Parameters:**
- `history` — training history from `train_model`
- `errors` — reconstruction error array
- `y_test` — ground-truth labels
- `threshold` — the best threshold from `find_best_threshold`
- `auc` — ROC-AUC score

**Plots produced:**

1. **Learning curve** — train loss and validation loss over epochs. A healthy curve shows both losses decreasing and converging. A large gap between them indicates overfitting.

2. **Reconstruction error distribution** — overlapping histograms of the reconstruction error for normal (blue) and fraud (red) samples, with the threshold drawn as a vertical dashed line. The better these two distributions are separated, the more effective the model is.

3. **ROC curve** — plots the true positive rate against the false positive rate at every possible threshold. The AUC (Area Under the Curve) summarises performance in a single number; 1.0 is perfect, 0.5 is random guessing.

**Note:** `matplotlib.use('Agg')` is set at the top of the file to allow saving plots in headless environments (no display attached).

---

### 5.9 `main`

```python
def main():
```

**Purpose:** Orchestrates the full pipeline end-to-end.

**Steps in order:**
1. Set random seeds for NumPy and TensorFlow for reproducibility
2. Download the dataset using `kagglehub`
3. Load the data
4. Preprocess: scale and split
5. Build the autoencoder and print the model summary
6. Train the model
7. Evaluate on the test set
8. Save the three diagnostic plots
9. Save the trained model as `autoencoder_fraud.keras`

---

## 6. Results Interpretation

The results below come from a real training run on the Kaggle credit card fraud dataset.

```
training done — final val loss: 0.2717
roc-auc score: 0.9379
best threshold (reconstruction error): 2.121386

--- classification report ---
              precision    recall  f1-score   support
      normal       1.00      1.00      1.00     56863
       fraud       0.68      0.80      0.74       492

    accuracy                           1.00     57355
   macro avg       0.84      0.90      0.87     57355
weighted avg       1.00      1.00      1.00     57355

confusion matrix:
  true  negative (normal correctly identified) : 56,681
  false positive (normal flagged as fraud)     : 182
  false negative (fraud missed)                : 100
  true  positive (fraud correctly identified)  : 392
```

---

### 6.1 Final Validation Loss

```
final val loss: 0.2717
```

This is the MSE between the original and reconstructed normal transactions on the held-out validation portion of the training set. A value of 0.2717 means the autoencoder can reproduce each feature of a normal transaction with an average squared error of about 0.27 — which is low given the inputs are standardised (mean 0, std 1) and values typically fall in the range of −3 to 3.

This confirms the model has successfully learned the structure of normal transactions.

---

### 6.2 ROC-AUC Score

```
roc-auc score: 0.9379
```

ROC-AUC measures how well the reconstruction error separates fraud from normal transactions across **all possible thresholds** — it is threshold-independent. A score of **0.9379** means that if you pick a random normal transaction and a random fraud transaction, there is a 93.79% chance the fraud transaction will have a higher reconstruction error. This is strong performance for an unsupervised approach trained with no fraud labels.

| AUC range | Interpretation |
|---|---|
| 0.50 | Random (no predictive power) |
| 0.70–0.80 | Acceptable |
| 0.80–0.90 | Good |
| 0.90–1.00 | Excellent |
| 1.00 | Perfect |

---

### 6.3 Threshold

```
best threshold: 2.121386
```

Any transaction whose reconstruction error exceeds **2.12** is classified as fraud. This value was selected automatically by searching for the threshold that maximises the F1-score for the fraud class on the test set.

The threshold sits far above the typical reconstruction error of normal transactions (centred around the validation loss of 0.27), which is exactly what is expected — the model struggles significantly more when asked to reconstruct fraud patterns.

---

### 6.4 Classification Report

#### Normal class (0)

```
precision: 1.00   recall: 1.00   f1-score: 1.00   support: 56,863
```

The model identifies virtually all normal transactions correctly. This is expected: the autoencoder was trained entirely on normal data and reconstructs it reliably. The 182 false positives (normal flagged as fraud) represent only 0.32% of all normal transactions — a very low false alarm rate.

#### Fraud class (1)

```
precision: 0.68   recall: 0.80   f1-score: 0.74   support: 492
```

- **Recall of 0.80** means the model correctly identified **392 out of 492** fraud cases. 100 fraudulent transactions went undetected (false negatives).
- **Precision of 0.68** means that of all transactions flagged as fraud, 68% were genuinely fraudulent and 32% were false alarms (normal transactions incorrectly flagged).
- **F1-score of 0.74** is the harmonic mean of precision and recall, balancing both concerns.

In fraud detection, **recall is typically prioritised** over precision: missing a fraud (false negative) is usually more costly than incorrectly flagging a legitimate transaction (false positive). A recall of 0.80 is therefore a meaningful result, though there is room for improvement.

---

### 6.5 Confusion Matrix

```
true  negative  (normal correctly identified) : 56,681
false positive  (normal flagged as fraud)     :    182
false negative  (fraud missed)                :    100
true  positive  (fraud correctly identified)  :    392
```

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actually Normal** | 56,681 ✓ | 182 ✗ |
| **Actually Fraud** | 100 ✗ | 392 ✓ |

- **56,681 true negatives** — the model lets these legitimate transactions through without raising an alert. Correct.
- **182 false positives** — legitimate transactions incorrectly flagged. These would require manual review or cause friction for customers.
- **100 false negatives** — fraudulent transactions the model missed entirely. These are the most serious errors in a real deployment.
- **392 true positives** — fraud cases successfully caught. This represents 79.7% of all 492 fraud cases in the test set.

---

## 7. Tradeoffs and Limitations

**Threshold sensitivity.** The threshold of 2.12 was chosen to maximise F1 on the test set. In production, the optimal threshold depends on the cost of each type of error. If missing fraud is ten times more costly than a false alarm, the threshold should be lowered to increase recall at the expense of precision.

**No fraud labels used in training.** This is both a strength and a limitation. The strength is that the model can detect novel fraud patterns it has never seen. The limitation is that the model cannot explicitly learn distinguishing features of fraud — it only knows what is normal.

**Concept drift.** Fraud patterns evolve over time. A model trained on 2013 transaction data may degrade as attackers change their behaviour. Periodic retraining on recent normal transactions is necessary for sustained performance.

**PCA features.** The V1–V28 features are already PCA-transformed. This means important variance has been extracted, which helps the autoencoder, but also means the individual features have no interpretable meaning.

**Class imbalance.** With only 492 fraud cases out of 284,807 transactions, even the test set is heavily imbalanced. Accuracy (99.7%) is a misleading metric here — it would be nearly as high if the model predicted "normal" for everything. Recall, precision, F1, and AUC are the correct metrics to focus on.
