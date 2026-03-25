# --------------------------------------------------------------
#  model_gru.py  –  Gated Recurrent Unit  (BPTT)
# --------------------------------------------------------------

import numpy as np
from config import HIDDEN_SIZE, NUM_LAYERS, SEED
from train import train, predict_year

NAME = "GRU"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def sigmoid_grad(s): return s * (1.0 - s)
def tanh_grad(t): return 1.0 - t * t


def init_weights() -> list:
    """
    Per layer: Wz Wr Wn (h,d)  Uz Ur Un (h,h)  bz br bn (h,)  — 9 arrays
    Output:    Wy (h,)  by (1,)
    """
    rng = np.random.default_rng(SEED)
    h = HIDDEN_SIZE
    ws = []
    for i in range(NUM_LAYERS):
        d = 1 if i == 0 else h
        s = np.sqrt(2.0 / (d + h))
        ws += [
            rng.normal(0, s, (h, d)),  # Wz
            rng.normal(0, s, (h, d)),  # Wr
            rng.normal(0, s, (h, d)),  # Wn
            rng.normal(0, s, (h, h)),  # Uz
            rng.normal(0, s, (h, h)),  # Ur
            rng.normal(0, s, (h, h)),  # Un
            np.zeros(h),               # bz
            np.zeros(h),               # br
            np.zeros(h),               # bn
        ]
    ws += [rng.normal(0, 0.1, h), np.zeros(1)]
    return [w.astype(np.float64) for w in ws]


def _unpack(weights, layer_idx):
    b = layer_idx * 9
    return (weights[b], weights[b+1], weights[b+2],
            weights[b+3], weights[b+4], weights[b+5],
            weights[b+6], weights[b+7], weights[b+8])


def _gru_fwd(x_seq, Wz, Wr, Wn, Uz, Ur, Un, bz, br, bn, h):
    T = len(x_seq)
    H = [np.zeros(h)]
    Z, R, N = [], [], []
    for t in range(T):
        x = x_seq[t] if x_seq.ndim > 1 else x_seq[t:t+1]
        z = sigmoid(Wz @ x + Uz @ H[t] + bz)
        r = sigmoid(Wr @ x + Ur @ H[t] + br)
        n = np.tanh (Wn @ x + Un @ (r * H[t]) + bn)
        hh = (1.0 - z) * H[t] + z * n
        Z.append(z); R.append(r); N.append(n); H.append(hh)
    return H, Z, R, N


def forward(x_seq: np.ndarray, weights: list) -> float:
    h = HIDDEN_SIZE
    seq = x_seq.astype(np.float64)
    for i in range(NUM_LAYERS):
        Wz,Wr,Wn,Uz,Ur,Un,bz,br,bn = _unpack(weights, i)
        H,Z,R,N = _gru_fwd(seq, Wz,Wr,Wn,Uz,Ur,Un,bz,br,bn, h)
        seq = np.array(H[1:])
    Wy, by = weights[-2], weights[-1]
    return float(np.dot(Wy, H[-1]) + by[0])


def forward_and_grad(x_seq: np.ndarray, y_true: float, weights: list):
    h = HIDDEN_SIZE
    T = len(x_seq)
    layers = []
    seq = x_seq.astype(np.float64)

    for i in range(NUM_LAYERS):
        Wz,Wr,Wn,Uz,Ur,Un,bz,br,bn = _unpack(weights, i)
        H,Z,R,N = _gru_fwd(seq, Wz,Wr,Wn,Uz,Ur,Un,bz,br,bn, h)
        layers.append((seq, H, Z, R, N))
        seq = np.array(H[1:])

    Wy, by = weights[-2], weights[-1]
    y_pred = float(np.dot(Wy, H[-1]) + by[0])
    loss = (y_pred - y_true) ** 2
    d_pred = 2.0 * (y_pred - y_true)

    dWy = d_pred * H[-1]
    dby = np.array([d_pred])
    dh = d_pred * Wy

    all_grads = []
    for i in range(NUM_LAYERS - 1, -1, -1):
        x_in, H, Z, R, N = layers[i]
        Wz,Wr,Wn,Uz,Ur,Un,bz,br,bn = _unpack(weights, i)

        dWz=np.zeros_like(Wz); dWr=np.zeros_like(Wr); dWn=np.zeros_like(Wn)
        dUz=np.zeros_like(Uz); dUr=np.zeros_like(Ur); dUn=np.zeros_like(Un)
        dbz=np.zeros_like(bz); dbr=np.zeros_like(br); dbn=np.zeros_like(bn)
        dx_seq = np.zeros_like(x_in)

        for t in range(T - 1, -1, -1):
            x = x_in[t] if x_in.ndim > 1 else x_in[t:t+1]

            # h = (1-z)*h_prev + z*n
            dn     = dh * Z[t]         * tanh_grad(N[t])
            dz_raw = dh * (N[t] - H[t]) * sigmoid_grad(Z[t])
            dh_prev = dh * (1.0 - Z[t])

            # n = tanh(Wn@x + Un@(r*h_prev) + bn)
            dr_h   = Un.T @ dn
            dr_raw = dr_h * H[t] * sigmoid_grad(R[t])
            dh_prev += dr_h * R[t]

            dWz += np.outer(dz_raw, x); dUz += np.outer(dz_raw, H[t]); dbz += dz_raw
            dWr += np.outer(dr_raw, x); dUr += np.outer(dr_raw, H[t]); dbr += dr_raw
            dWn += np.outer(dn, x);     dUn += np.outer(dn, R[t] * H[t]); dbn += dn
            dh_prev += Uz.T @ dz_raw + Ur.T @ dr_raw

            dx = Wz.T @ dz_raw + Wr.T @ dr_raw + Wn.T @ dn
            dx_seq[t] = dx if x_in.ndim > 1 else dx[0]
            dh = dh_prev

        all_grads.insert(0, [dWz,dWr,dWn, dUz,dUr,dUn, dbz,dbr,dbn])

    flat = []
    for lg in all_grads: flat.extend(lg)
    flat += [dWy, dby]
    return loss, flat


def run(X: np.ndarray, y: np.ndarray, seed_seq: np.ndarray) -> dict:
    weights = init_weights()
    losses  = train(forward_and_grad, weights, X, y, NAME)
    preds   = predict_year(forward, weights, seed_seq)
    return {"name": NAME, "losses": losses, "preds_2025": preds}

# --------------------|CONVENIENCE WRAPPERS|-----------------------------------
from train import train as _train, predict_year as _predict_year

def train_model(weights: list, X, y) -> list:
    #train in-place and return losses list
    return _train(forward_and_grad, weights, X, y, NAME)

def predict_year_from_weights(weights: list, seed_seq) -> list:
    #run autoregressive inference for PREDICT_YEAR
    return _predict_year(forward, weights, seed_seq)