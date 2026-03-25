# --------------------------------------------------------------
#  model_bilstm.py  –  Bidirectional LSTM  (BPTT)
# --------------------------------------------------------------

import numpy as np
from config import HIDDEN_SIZE, NUM_LAYERS, SEED
from train import train, predict_year

NAME = "Bi-LSTM"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def sigmoid_grad(s): return s * (1.0 - s)
def tanh_grad(t): return 1.0 - t * t


def init_weights() -> list:
    """
    forward  layers: NUM_LAYERS × 12 arrays  (same layout as model_lstm)
    backward layers: NUM_LAYERS × 12 arrays
    output:  Wy (2h,)  by (1,)              (concat fwd+bwd final states)
    """
    rng = np.random.default_rng(SEED)
    h = HIDDEN_SIZE
    ws = []
    for _ in range(2):           # two directions
        for i in range(NUM_LAYERS):
            d = 1 if i == 0 else h
            s = np.sqrt(2.0 / (d + h))
            ws += [
                rng.normal(0, s, (h, d)),  # Wi
                rng.normal(0, s, (h, d)),  # Wf
                rng.normal(0, s, (h, d)),  # Wg
                rng.normal(0, s, (h, d)),  # Wo
                rng.normal(0, s, (h, h)),  # Ui
                rng.normal(0, s, (h, h)),  # Uf
                rng.normal(0, s, (h, h)),  # Ug
                rng.normal(0, s, (h, h)),  # Uo
                np.zeros(h),               # bi
                np.ones(h),                # bf
                np.zeros(h),               # bg
                np.zeros(h),               # bo
            ]
    ws += [rng.normal(0, 0.1, 2 * h), np.zeros(1)]   # Wy, by
    return [w.astype(np.float64) for w in ws]


def _unpack(weights, base):
    return (weights[base],   weights[base+1], weights[base+2],  weights[base+3],
            weights[base+4], weights[base+5], weights[base+6],  weights[base+7],
            weights[base+8], weights[base+9], weights[base+10], weights[base+11])


def _lstm_fwd(x_seq, Wi, Wf, Wg, Wo, Ui, Uf, Ug, Uo, bi, bf, bg, bo, h):
    T = len(x_seq)
    H = [np.zeros(h)]; C = [np.zeros(h)]
    I, F, G, O = [], [], [], []
    for t in range(T):
        x   = x_seq[t] if x_seq.ndim > 1 else x_seq[t:t+1]
        i_g = sigmoid(Wi @ x + Ui @ H[t] + bi)
        f_g = sigmoid(Wf @ x + Uf @ H[t] + bf)
        g_g = np.tanh (Wg @ x + Ug @ H[t] + bg)
        o_g = sigmoid(Wo @ x + Uo @ H[t] + bo)
        c   = f_g * C[t] + i_g * g_g
        hh  = o_g * np.tanh(c)
        I.append(i_g); F.append(f_g); G.append(g_g); O.append(o_g)
        H.append(hh); C.append(c)
    return H, C, I, F, G, O


def _lstm_bwd_grads(x_in, H, C, I, F, G, O,
                    Wi, Wf, Wg, Wo, Ui, Uf, Ug, Uo,
                    bi, bf, bg, bo, dh_last, h):
    """BPTT for one LSTM layer. Returns weight grads + dx_seq."""
    T = len(x_in)
    dWi=np.zeros_like(Wi); dWf=np.zeros_like(Wf)
    dWg=np.zeros_like(Wg); dWo=np.zeros_like(Wo)
    dUi=np.zeros_like(Ui); dUf=np.zeros_like(Uf)
    dUg=np.zeros_like(Ug); dUo=np.zeros_like(Uo)
    dbi=np.zeros_like(bi); dbf=np.zeros_like(bf)
    dbg=np.zeros_like(bg); dbo=np.zeros_like(bo)

    dh  = dh_last
    dc  = np.zeros(h)
    dx_seq = np.zeros_like(x_in)

    for t in range(T - 1, -1, -1):
        x      = x_in[t] if x_in.ndim > 1 else x_in[t:t+1]
        tanh_c = np.tanh(C[t+1])

        do_raw = (dh * tanh_c) * sigmoid_grad(O[t])
        dc += (dh * O[t]) * tanh_grad(tanh_c)
        di_raw = (dc * G[t]) * sigmoid_grad(I[t])
        df_raw = (dc * C[t]) * sigmoid_grad(F[t])
        dg_raw = (dc * I[t]) * tanh_grad(G[t])
        dc = dc * F[t]

        dWi += np.outer(di_raw, x);  dUi += np.outer(di_raw, H[t]); dbi += di_raw
        dWf += np.outer(df_raw, x);  dUf += np.outer(df_raw, H[t]); dbf += df_raw
        dWg += np.outer(dg_raw, x);  dUg += np.outer(dg_raw, H[t]); dbg += dg_raw
        dWo += np.outer(do_raw, x);  dUo += np.outer(do_raw, H[t]); dbo += do_raw

        dh = Ui.T @ di_raw + Uf.T @ df_raw + Ug.T @ dg_raw + Uo.T @ do_raw
        dx = Wi.T @ di_raw + Wf.T @ df_raw + Wg.T @ dg_raw + Wo.T @ do_raw
        dx_seq[t] = dx if x_in.ndim > 1 else dx[0]

    w_grads = [dWi,dWf,dWg,dWo, dUi,dUf,dUg,dUo, dbi,dbf,dbg,dbo]
    return w_grads, dx_seq


def forward(x_seq: np.ndarray, weights: list) -> float:
    h = HIDDEN_SIZE
    offset = NUM_LAYERS * 12   #where backward layers start
    seq = x_seq.astype(np.float64)

    # Forward direction
    fwd_H = None
    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, i*12)
        fwd_H,_,_,_,_,_ = _lstm_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        seq = np.array(fwd_H[1:])
    h_fwd = fwd_H[-1]

    # Backward direction (reversed sequence)
    seq = x_seq[::-1].astype(np.float64)
    bwd_H = None
    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, offset + i*12)
        bwd_H,_,_,_,_,_ = _lstm_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        seq = np.array(bwd_H[1:])
    h_bwd = bwd_H[-1]

    Wy, by = weights[-2], weights[-1]
    return float(np.dot(Wy, np.concatenate([h_fwd, h_bwd])) + by[0])


def forward_and_grad(x_seq: np.ndarray, y_true: float, weights: list):
    h = HIDDEN_SIZE
    offset = NUM_LAYERS * 12
    seq_f = x_seq.astype(np.float64)
    seq_b = x_seq[::-1].astype(np.float64)

    # |FORWARD CACHE|------------------------------------------------
    fwd_cache, bwd_cache = [], []

    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, i*12)
        H,C,I,F,G,O = _lstm_fwd(seq_f, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        fwd_cache.append((seq_f, H, C, I, F, G, O))
        seq_f = np.array(H[1:])
    h_fwd = H[-1]

    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, offset + i*12)
        H,C,I,F,G,O = _lstm_fwd(seq_b, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        bwd_cache.append((seq_b, H, C, I, F, G, O))
        seq_b = np.array(H[1:])
    h_bwd = H[-1]

    Wy, by = weights[-2], weights[-1]
    y_pred = float(np.dot(Wy, np.concatenate([h_fwd, h_bwd])) + by[0])
    loss = (y_pred - y_true) ** 2
    d_pred = 2.0 * (y_pred - y_true)

    dWy = d_pred * np.concatenate([h_fwd, h_bwd])
    dby = np.array([d_pred])
    dh_fwd_last = d_pred * Wy[:h]
    dh_bwd_last = d_pred * Wy[h:]

    # |BACKWARD THROUGH FORWARD LAYERS|---------------------------
    fwd_layer_grads = []
    dh = dh_fwd_last
    for i in range(NUM_LAYERS - 1, -1, -1):
        x_in,H,C,I,F,G,O = fwd_cache[i]
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, i*12)
        lg, _ = _lstm_bwd_grads(x_in, H, C, I, F, G, O,
                                Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo, bi,bf,bg,bo, dh, h)
        fwd_layer_grads.insert(0, lg)
        dh = _

    # |BACKWARD THROUGH BACKWARD LAYERS|--------------------------
    bwd_layer_grads = []
    dh = dh_bwd_last
    for i in range(NUM_LAYERS - 1, -1, -1):
        x_in,H,C,I,F,G,O = bwd_cache[i]
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, offset + i*12)
        lg, _ = _lstm_bwd_grads(x_in, H, C, I, F, G, O,
                                Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo, bi,bf,bg,bo, dh, h)
        bwd_layer_grads.insert(0, lg)
        dh = _

    flat = []
    for lg in fwd_layer_grads: flat.extend(lg)
    for lg in bwd_layer_grads: flat.extend(lg)
    flat += [dWy, dby]

    return loss, flat


def run(X: np.ndarray, y: np.ndarray, seed_seq: np.ndarray) -> dict:
    weights = init_weights()
    losses  = train(forward_and_grad, weights, X, y, NAME)
    preds   = predict_year(forward, weights, seed_seq)
    return {"name": NAME, "losses": losses, "preds_2025": preds}

# ── convenience wrappers used by main.py ─────────────────────
from train import train as _train, predict_year as _predict_year

def train_model(weights: list, X, y) -> list:
    """Train in-place and return losses list."""
    return _train(forward_and_grad, weights, X, y, NAME)

def predict_year_from_weights(weights: list, seed_seq) -> list:
    """Run autoregressive inference for PREDICT_YEAR."""
    return _predict_year(forward, weights, seed_seq)