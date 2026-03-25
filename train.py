# --------------------------------------------------------------
#  train.py  –  training loop, Adam optimiser, and inference
# --------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from config import EPOCHS, LEARNING_RATE, CLIP_GRAD, SEQ_LEN


#-------------------------|ADAM STATE|------------------------------------

def make_adam_state(weights: list) -> dict:
    #initialise first/second moment vectors for every weight array
    return {
        "m": [np.zeros_like(w) for w in weights],
        "v": [np.zeros_like(w) for w in weights],
        "t": 0,
    }


def adam_step(weights: list, grads: list, state: dict, lr: float = LEARNING_RATE, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    #in-place Adam update for all weight arrays
    state["t"] += 1
    t = state["t"]
    for i, (w, g) in enumerate(zip(weights, grads)):
        state["m"][i] = beta1 * state["m"][i] + (1 - beta1) * g
        state["v"][i] = beta2 * state["v"][i] + (1 - beta2) * g * g
        m_hat = state["m"][i] / (1 - beta1 ** t)
        v_hat = state["v"][i] / (1 - beta2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)


def clip_grads(grads: list, max_norm: float = CLIP_GRAD) -> list:
    #global gradient norm clipping
    total_norm = np.sqrt(sum(np.sum(g * g) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        grads = [g * scale for g in grads]
    return grads


# ----------------------|TRAINING LOOP|-----------------------

def train(forward_and_grad_fn, weights: list,
          X: np.ndarray, y: np.ndarray, name: str) -> list:
    """
    Train using analytical gradients (BPTT) + Adam.
    Args:
        forward_and_grad_fn – callable(x, y_true, weights)
                              returns (loss: float, grads: list[np.ndarray])
        weights – list of numpy arrays, updated in-place
        X, y – (N, SEQ_LEN) inputs and (N,) targets
        name – label shown on the progress bar

    Returns:
        losses – list of mean MSE per epoch
    """
    n     = len(X)
    state = make_adam_state(weights)
    losses = []

    bar = tqdm(range(EPOCHS), desc=f"{name:<22}", unit="ep", ncols=88, leave=True)

    for _ in bar:
        idx = np.random.permutation(n)
        epoch_loss = 0.0

        for i in idx:
            loss, grads = forward_and_grad_fn(X[i], y[i], weights)
            grads = clip_grads(grads)
            adam_step(weights, grads, state)
            epoch_loss += loss

        avg = epoch_loss / n
        losses.append(avg)
        bar.set_postfix(MSE=f"{avg:.5f}")

    return losses


# ------------------|INFERENCE|------------------------

def predict_year(forward_fn, weights: list, seed_seq: np.ndarray) -> list:
    
    #autoregressively predict 12 months.
    #seed_seq: normalised training series (uses last SEQ_LEN values as seed).
    
    buf = list(seed_seq[-SEQ_LEN:])
    out = []
    for _ in range(12):
        x = np.array(buf[-SEQ_LEN:], dtype=np.float64)
        p = forward_fn(x, weights)
        out.append(float(p))
        buf.append(p)
    return out