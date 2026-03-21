"""
KOUKOSIAS ATHANASIOS-UTH-2026
Dataset:https://www.kaggle.com/code/quadeer15sh/flickr8k-image-captioning-using-cnns-lstmsx
"""

import os, re, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend for WSL
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ----------------|CONFIG|----------------

IMAGE_DIR="flickr8k/Images"
CAPTIONS_FILE="flickr8k/captions.txt"
FEATURES_FILE="image_features_spatial.pkl"
TOKENIZER_FILE="tokenizer.pkl"

IMAGE_SIZE=(224, 224)
EMBED_DIM=256
LSTM_UNITS=512
ATTENTION_DIM=256
DENSE_UNITS=512
DROPOUT_RATE=0.3
VOCAB_SIZE=10000
MAX_LEN=22
BATCH_SIZE=64
EPOCHS=30
TRAIN_SPLIT=0.90

np.random.seed(42)
tf.random.set_seed(42)

# ----------------|CAPTIONS|----------------

def load_captions(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    captions = {}
    for _, row in df.iterrows():
        captions.setdefault(str(row["image"]).strip(), []).append(str(row["caption"]).strip())
    return captions

def clean(text):
    text = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    return f"<start> {text} <end>"

def clean_all(captions):
    return {img: [clean(c) for c in caps] for img, caps in captions.items()}

# ----------------|VOCABULARY|----------------

def build_vocab(captions):
    counter  = Counter(w for caps in captions.values() for c in caps for w in c.split())
    special  = ["<pad>", "<unk>", "<start>", "<end>"]
    vocab    = special + [w for w, _ in counter.most_common(VOCAB_SIZE) if w not in special]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

def to_seq(caption, word2idx):
    return [word2idx.get(w, word2idx["<unk>"]) for w in caption.split()]

# ----------------|CNN FEATURE EXTRACTION|----------------
#ResNet50 (frozen, no top, no pooling) extracts a 7x7 spatial grid per image
#each image becomes 49 region vectors of 2048 dims, saved to disk for reuse.


def build_cnn():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    return base

def load_img(path):
    return preprocess_input(
        np.array(Image.open(path).convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
    )

def extract_features(cnn):
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, "rb") as f:
            return pickle.load(f)

    files    = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    features = {}

    for i in tqdm(range(0, len(files), 32), desc="Extracting CNN features"):
        batch = files[i:i+32]
        imgs, names = [], []
        for fname in batch:
            try:
                imgs.append(load_img(os.path.join(IMAGE_DIR, fname)))
                names.append(fname)
            except Exception:
                pass
        if imgs:
            #CNN forward pass: (batch, 224, 224, 3) → (batch, 7, 7, 2048)
            feats = cnn.predict(np.stack(imgs), verbose=0)
            #flatten spatial grid: (batch, 7, 7, 2048) → (batch, 49, 2048)
            feats = feats.reshape(len(imgs), -1, feats.shape[-1])
            features.update(zip(names, feats))

    with open(FEATURES_FILE, "wb") as f:
        pickle.dump(features, f)
    return features

# ----------------|GENERATOR|----------------

class CaptionGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, captions, features, word2idx):
        self.features = features
        self.samples  = [
            (img_id, seq[:t], seq[t])
            for img_id in ids if img_id in features
            for cap    in captions.get(img_id, [])
            for seq    in [to_seq(cap, word2idx)]
            for t      in range(1, len(seq))
        ]
        np.random.shuffle(self.samples)

    def __len__(self):
        return max(1, len(self.samples) // BATCH_SIZE)

    def __getitem__(self, idx):
        batch                  = self.samples[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
        img_ids, seqs, targets = zip(*batch)
        imgs    = np.array([self.features[i] for i in img_ids], dtype=np.float32)
        caps    = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
        targets = np.array(targets, dtype=np.int32)
        return (imgs, caps), targets

    def on_epoch_end(self):
        np.random.shuffle(self.samples)

# ----------------|ATTENTION|----------------
#Bahdanau (additive) attention over the 49 CNN spatial regions
#at each decode step the LSTM hidden state queries the image regions and returns a weighted context vector — the model's visual focus.


class BahdanauAttention(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim      = dim
        self.W_feat   = Dense(dim, use_bias=False)  # projects image regions
        self.W_hidden = Dense(dim, use_bias=False)  # projects LSTM hidden state
        self.V        = Dense(1,   use_bias=False)  # scalar score per region

    def call(self, features, hidden):
        #features: (batch, 49, EMBED_DIM)
        #hidden: (batch, LSTM_UNITS) --> expanded --> (batch, 1, LSTM_UNITS)
        score   = tf.nn.tanh(self.W_feat(features) + self.W_hidden(tf.expand_dims(hidden, 1)))
        weights = tf.squeeze(tf.nn.softmax(self.V(score), axis=1), -1)  #(batch, 49)
        context = tf.reduce_sum(features * tf.expand_dims(weights, -1), axis=1)  #(batch, EMBED_DIM)
        return context, weights

    def get_config(self):
        return {**super().get_config(), "dim": self.dim}


# ----------------|MODEL: CNN BRANCH + LSTM BRANCH + ATTENTION + CLASSIFIER|----------------


def build_model(num_regions, feature_dim, vocab_size):

    # ----------------|CNN BRANCH|----------------
    #pre-extracted ResNet50 spatial features enter here.
    #each image is represented as 49 region vectors projected to EMBED_DIM.
    #input:  (batch, 49, 2048)
    #output: (batch, 49, 256)
    # -------------------------------------------------------------------------
    img_input = Input(shape=(num_regions, feature_dim), name="image_features")
    img_drop  = Dropout(DROPOUT_RATE)(img_input)
    img_proj  = Dense(EMBED_DIM, activation="relu", name="cnn_projection")(img_drop)
    #img_proj: (batch, 49, 256) --> 49 spatial regions ready for attention


    # ----------------|LSTM BRANCH|----------------
    #encodes the partial caption into a hidden state that drives attention
    #end word prediction.
    #input:  (batch, MAX_LEN)   --> token indices
    #output: (batch, 512) -->final LSTM hidden state
    # -------------------------------------------------------------------------
    cap_input = Input(shape=(MAX_LEN,), name="partial_caption")

    #embedding: token indices --> dense vectors
    embedding = Embedding(vocab_size, EMBED_DIM, mask_zero=True, name="token_embedding")(cap_input)
    embedding = Dropout(DROPOUT_RATE)(embedding)
    #embedding: (batch, MAX_LEN, 256)

    #LSTM layer 1 - passes full sequence to layer 2
    lstm1_out = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_1")(embedding)
    lstm1_out = Dropout(DROPOUT_RATE)(lstm1_out)
    #lstm1_out: (batch, MAX_LEN, 512)

    #LSTM layer 2 - produces the final hidden state used as attention query
    lstm2_out, hidden_state, _ = LSTM(
        LSTM_UNITS, return_sequences=False, return_state=True, name="lstm_2"
    )(lstm1_out)
    lstm2_out = Dropout(DROPOUT_RATE)(lstm2_out)
    #lstm2_out / hidden_state: (batch, 512)


    # -------------------------------------------------------------------------
    # ATTENTION: bridges CNN and LSTM
    #query : lstm2_out  (batch, 512): what the decoder wants to know
    #keys  : img_proj   (batch, 49, 256): what each image region contains
    #output: context    (batch, 256): weighted sum of attended regions
    # -------------------------------------------------------------------------
    context, _ = BahdanauAttention(ATTENTION_DIM, name="attention")(img_proj, lstm2_out)
    #context: (batch, 256)


    # -------------------------------------------------------------------------
    # CLASSIFIER HEAD
    #fuses LSTM language context and attended image context to predict next word.
    # -------------------------------------------------------------------------
    merged = Concatenate(name="merge_lstm_context")([lstm2_out, context])
    #merged: (batch, 768) :  512 LSTM + 256 attention context

    hidden = Dense(DENSE_UNITS, activation="relu", name="classifier_hidden")(merged)
    hidden = Dropout(DROPOUT_RATE)(hidden)
    output = Dense(vocab_size, activation="softmax", name="word_prediction")(hidden)
    #output: (batch, vocab_size)


# ----------------|COMPILE|----------------
    model = Model(inputs=[img_input, cap_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return model

# ----------------|INFERENCE|----------------

def greedy(feat, model, word2idx, idx2word):
    seq = [word2idx["<start>"]]
    img = np.expand_dims(feat, 0)
    for _ in range(MAX_LEN):
        probs   = model.predict([img, pad_sequences([seq], maxlen=MAX_LEN, padding="post")], verbose=0)[0]
        next_id = int(np.argmax(probs))
        if idx2word.get(next_id) == "<end>":
            break
        seq.append(next_id)
    return " ".join(idx2word.get(i, "<unk>") for i in seq[1:])

def beam_search(feat, model, word2idx, idx2word, width=5):
    start, end  = word2idx["<start>"], word2idx["<end>"]
    img         = np.expand_dims(feat, 0)
    beams, done = [(0.0, [start])], []
    for _ in range(MAX_LEN):
        candidates = []
        for score, seq in beams:
            if seq[-1] == end:
                done.append((score, seq)); continue
            probs = model.predict([img, pad_sequences([seq], maxlen=MAX_LEN, padding="post")], verbose=0)[0]
            candidates += [(score + np.log(probs[w] + 1e-10), seq + [int(w)]) for w in np.argsort(probs)[-width:]]
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:width]
    best  = max(done + beams, key=lambda x: x[0] / max(len(x[1]), 1))
    return " ".join(idx2word[i] for i in best[1][1:] if idx2word.get(i) not in ("<end>", "<start>", None))

# ----------------|EVALUATION|----------------
def evaluate_bleu(val_ids, captions, features, model, word2idx, idx2word):
    smooth = SmoothingFunction().method1
    refs, hyps = [], []
    for img_id in tqdm(val_ids[:500], desc="BLEU"):
        if img_id not in features: continue
        hyps.append(greedy(features[img_id], model, word2idx, idx2word).split())
        refs.append([clean(c).replace("<start> ", "").replace(" <end>", "").split()
                     for c in captions[img_id]])
    scores = [corpus_bleu(refs, hyps, weights=w, smoothing_function=smooth)
              for w in [(1,0,0,0), (.5,.5,0,0), (.33,.33,.34,0), (.25,.25,.25,.25)]]
    print(f"BLEU  1:{scores[0]:.4f}  2:{scores[1]:.4f}  3:{scores[2]:.4f}  4:{scores[3]:.4f}")
    return scores

# ----------------|PLOTS|----------------

def save_training_curves(history):
    plt.figure(figsize=(10, 4))
    for i, (key, title) in enumerate([("loss", "Loss"), ("accuracy", "Accuracy")]):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history[key], label="train")
        plt.plot(history.history[f"val_{key}"],  label="val")
        plt.title(title); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.close()
    print("saved training_curves.png")

def save_example_captions(val_ids, captions, features, model, word2idx, idx2word, n=5):
    sample_ids = [i for i in val_ids if i in features][:n]
    fig, axes  = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    for ax, img_id in zip(axes, sample_ids):
        try:
            img = Image.open(os.path.join(IMAGE_DIR, img_id)).convert("RGB")
        except FileNotFoundError:
            ax.axis("off"); continue
        pred = greedy(features[img_id], model, word2idx, idx2word)
        ax.imshow(img)
        ax.set_title(pred, fontsize=8, wrap=True)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("caption_examples.png", dpi=120)
    plt.close()
    print("saved caption_examples.png")



# ----------------|MAIN|----------------
def main():
    captions = clean_all(load_captions(CAPTIONS_FILE))
    word2idx, idx2word = build_vocab(captions)
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump((word2idx, idx2word), f)

    features = extract_features(build_cnn())
    num_regions, fdim = next(iter(features.values())).shape

    all_ids = list(features.keys())
    np.random.shuffle(all_ids)
    split = int(len(all_ids) * TRAIN_SPLIT)
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    train_gen = CaptionGenerator(train_ids, captions, features, word2idx)
    val_gen = CaptionGenerator(val_ids,   captions, features, word2idx)

    model = build_model(num_regions, fdim, len(word2idx))
    model.summary()

    history = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True),
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ]
    )

    save_training_curves(history)
    evaluate_bleu(val_ids, captions, features, model, word2idx, idx2word)
    save_example_captions(val_ids, captions, features, model, word2idx, idx2word)

    demo = val_ids[0]
    print(f"\nimage  : {demo}")
    print(f"greedy : {greedy(features[demo], model, word2idx, idx2word)}")
    print(f"beam   : {beam_search(features[demo], model, word2idx, idx2word)}")
    print(f"truth  : {captions[demo][0]}")


if __name__ == "__main__":
    main()
