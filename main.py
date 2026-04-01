# dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#
# how it works:
# - train the autoencoder only on normal transactions
# - it learns to reconstruct normal transactions well
# - fraud transactions have high reconstruction error -> detected as anomalies

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras


# --------------------------------------------------------------
# load_data: reads the csv and prints basic dataset statistics
# --------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(f"{path}/creditcard.csv")
    print(f"total samples : {len(df):,}")
    print(f"normal: {(df['Class'] == 0).sum():,}")
    print(f"fraud: {(df['Class'] == 1).sum():,}")
    print(f"fraud rate: {df['Class'].mean() * 100:.3f}%")
    return df


# --------------------------------------------------------------
# preprocess: scales features and splits into train/test sets
# train set contains only normal transactions (unsupervised)
# test set contains both normal and fraud transactions
# --------------------------------------------------------------
def preprocess(df):
    features = [c for c in df.columns if c != "Class"]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df["Class"].values

    X_normal = X[y == 0]
    X_fraud = X[y == 1]

    # train on normal only, validate on held-out normal + all fraud
    X_train, X_val_normal = train_test_split(X_normal, test_size=0.2, random_state=42)

    X_test = np.concatenate([X_val_normal, X_fraud])
    y_test = np.concatenate([np.zeros(len(X_val_normal)), np.ones(len(X_fraud))])

    print(f"train set : {X_train.shape[0]:,} normal transactions")
    print(f"test set  : {X_test.shape[0]:,} transactions ({len(X_val_normal):,} normal + {len(X_fraud):,} fraud)")

    return X_train, X_test, y_test, X.shape[1]


# --------------------------------------------------------------
#build_autoencoder: creates a symmetric autoencoder network
#encoder: input_dim -> 32 -> 16 -> 8 (bottleneck)
#decoder: 8 -> 16 -> 32 -> input_dim
# --------------------------------------------------------------
def build_autoencoder(input_dim):
    inputs = keras.Input(shape=(input_dim,), name="input")

    # encoder
    x = keras.layers.Dense(32, activation="relu", name="enc_32")(inputs)
    x = keras.layers.Dense(16, activation="relu", name="enc_16")(x)
    encoded = keras.layers.Dense(8, activation="relu", name="bottleneck")(x)

    # decoder
    x = keras.layers.Dense(16, activation="relu", name="dec_16")(encoded)
    x = keras.layers.Dense(32, activation="relu", name="dec_32")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="output")(x)

    model = keras.Model(inputs, outputs, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# --------------------------------------------------------------
#train_model: trains the autoencoder on normal transactions only
#uses early stopping to avoid overfitting
# --------------------------------------------------------------
def train_model(model, X_train, epochs=30, batch_size=256):
    print("training autoencoder")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, X_train,   #input = target (reconstruction task)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    final_loss = history.history["val_loss"][-1]
    print(f"training done - final val loss: {final_loss:.4f}")
    return history


# --------------------------------------------------------------
# compute_reconstruction_error: calculates per-sample mse
# high error = likely fraud (autoencoder fails to reconstruct it)
# --------------------------------------------------------------
def compute_reconstruction_error(model, X):
    X_reconstructed = model.predict(X, verbose=0)
    errors = np.mean(np.square(X - X_reconstructed), axis=1)
    return errors


# --------------------------------------------------------------
# find_best_threshold: scans candidate thresholds and picks the one
# that maximizes f1-score for the fraud class
# --------------------------------------------------------------
def find_best_threshold(errors, y_true):
    thresholds = np.percentile(errors, np.arange(80, 100, 0.5))
    best_thresh, best_f1 = 0, 0

    for t in thresholds:
        y_pred = (errors >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh


# --------------------------------------------------------------
#evaluate: computes roc-auc, finds best threshold, prints report
# --------------------------------------------------------------
def evaluate(model, X_test, y_test):
    errors = compute_reconstruction_error(model, X_test)

    auc = roc_auc_score(y_test, errors)
    print(f"roc-auc score: {auc:.4f}")

    threshold = find_best_threshold(errors, y_test)
    print(f"best threshold (reconstruction error): {threshold:.6f}")

    y_pred = (errors >= threshold).astype(int)

    print("\n--- classification report ---")
    print(classification_report(y_test, y_pred, target_names=["normal", "fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    print("confusion matrix:")
    print(f"true  negative (normal correctly identified): {cm[0,0]:,}")
    print(f"false positive (normal flagged as fraud): {cm[0,1]:,}")
    print(f"false negative (fraud missed): {cm[1,0]:,}")
    print(f"true  positive (fraud correctly identified): {cm[1,1]:,}")

    return errors, threshold, auc, y_pred


# --------------------------------------------------------------
#plot_results: saves three plots to results.png
#1. learning curve  2. error distribution  3. roc curve
# --------------------------------------------------------------
def plot_results(history, errors, y_test, threshold, auc):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("autoencoder - credit card fraud detection", fontsize=14, fontweight="bold")

    #learning curve
    ax = axes[0]
    ax.plot(history.history["loss"], label="train loss")
    ax.plot(history.history["val_loss"], label="val loss", linestyle="--")
    ax.set_title("learning curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mse loss")
    ax.legend()
    ax.grid(alpha=0.3)

    #reconstruction error distribution
    ax = axes[1]
    bins = np.linspace(0, np.percentile(errors, 99.5), 100)
    ax.hist(errors[y_test == 0], bins=bins, alpha=0.6, label="normal", color="steelblue", density=True)
    ax.hist(errors[y_test == 1], bins=bins, alpha=0.6, label="fraud",  color="tomato",    density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    ax.set_title("reconstruction error distribution")
    ax.set_xlabel("mse")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    #roc curve
    ax = axes[2]
    fpr, tpr, _ = roc_curve(y_test, errors)
    ax.plot(fpr, tpr, color="darkorange", label=f"roc (auc={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("roc curve")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("plots saved to results.png")


# --------------------------------------------------------------
#main: runs the full pipeline end to end
# --------------------------------------------------------------
def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    #1. load data
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = load_data(path)

    #2. preprocess
    X_train, X_test, y_test, input_dim = preprocess(df)

    #3. build model
    model = build_autoencoder(input_dim)
    model.summary()

    #4. train
    history = train_model(model, X_train, epochs=30, batch_size=256)

    #5. evaluate
    errors, threshold, auc, y_pred = evaluate(model, X_test, y_test)

    #6. plot
    plot_results(history, errors, y_test, threshold, auc)

    # 7. save model
    model.save("autoencoder_fraud.keras")
    print("model saved to autoencoder_fraud.keras")


if __name__ == "__main__":
    main()