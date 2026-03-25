# --------------------------------------------------------------
#  model_convlstm.py  –  Conv1D → LSTM  (BPTT)
#
#  A single 1-D conv filter (kernel=3) extracts local temporal
#  patterns from the input, then feeds into a stacked LSTM.
# --------------------------------------------------------------

import numpy as np
from config import HIDDEN_SIZE, NUM_LAYERS, SEED
from train import train, predict_year

NAME = "ConvLSTM"
CONV_K = 3

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def sigmoid_grad(s): return s * (1.0 - s)
def tanh_grad(t): return 1.0 - t * t


def init_weights() -> list:
    """
    [Wc (CONV_K,)  bc (1,)
     LSTM layer 0 (12 arrays) ... LSTM layer N
     Wy (h,)  by (1,)]
    """
    rng = np.random.default_rng(SEED)
    h = HIDDEN_SIZE
    ws = [rng.normal(0, 0.1, CONV_K), np.zeros(1)]   #conv
    for i in range(NUM_LAYERS):
        d = 1 if i == 0 else h
        s = np.sqrt(2.0 / (d + h))
        ws += [
            rng.normal(0, s, (h, d)), rng.normal(0, s, (h, d)),  # Wi Wf
            rng.normal(0, s, (h, d)), rng.normal(0, s, (h, d)),  # Wg Wo
            rng.normal(0, s, (h, h)), rng.normal(0, s, (h, h)),  # Ui Uf
            rng.normal(0, s, (h, h)), rng.normal(0, s, (h, h)),  # Ug Uo
            np.zeros(h), np.ones(h), np.zeros(h), np.zeros(h),   # bi bf bg bo
        ]
    ws += [rng.normal(0, 0.1, h), np.zeros(1)]
    return [w.astype(np.float64) for w in ws]


def _unpack_lstm(weights, layer_idx):
    base = 2 + layer_idx * 12   # skip Wc, bc
    return (weights[base],   weights[base+1], weights[base+2],  weights[base+3],
            weights[base+4], weights[base+5], weights[base+6],  weights[base+7],
            weights[base+8], weights[base+9], weights[base+10], weights[base+11])


def _conv1d_fwd(x_seq, Wc, bc):
    #same-pad 1-D conv, single filter. Returns (T,) output + padded input
    T = len(x_seq)
    pad = CONV_K // 2
    padded = np.pad(x_seq, pad)
    out = np.array([np.dot(Wc, padded[t:t+CONV_K]) + bc[0] for t in range(T)])
    return out, padded


def _conv1d_bwd(d_out, padded, Wc):
    #backprop through same-pad conv1d
    T = len(d_out)
    dWc = np.zeros_like(Wc)
    dbc = np.array([d_out.sum()])
    d_pad = np.zeros_like(padded)
    pad = CONV_K // 2
    for t in range(T):
        dWc += d_out[t] * padded[t:t+CONV_K]
        d_pad[t:t+CONV_K] += d_out[t] * Wc
    d_x = d_pad[pad:pad+T]
    return dWc, dbc, d_x


def _lstm_fwd(x_seq, Wi, Wf, Wg, Wo, Ui, Uf, Ug, Uo, bi, bf, bg, bo, h):
    T = len(x_seq)
    H = [np.zeros(h)]; C = [np.zeros(h)]
    I, F, G, O = [], [], [], []
    for t in range(T):
        x = x_seq[t:t+1] if x_seq.ndim == 1 else x_seq[t]
        i_g = sigmoid(Wi @ x + Ui @ H[t] + bi)
        f_g = sigmoid(Wf @ x + Uf @ H[t] + bf)
        g_g = np.tanh (Wg @ x + Ug @ H[t] + bg)
        o_g = sigmoid(Wo @ x + Uo @ H[t] + bo)
        c = f_g * C[t] + i_g * g_g
        hh = o_g * np.tanh(c)
        I.append(i_g); F.append(f_g); G.append(g_g); O.append(o_g)
        H.append(hh); C.append(c)
    return H, C, I, F, G, O


def _lstm_bwd(x_in, H, C, I, F, G, O,
              Wi, Wf, Wg, Wo, Ui, Uf, Ug, Uo,
              bi, bf, bg, bo, dh_last, h):
    T = len(x_in)
    dWi=np.zeros_like(Wi); dWf=np.zeros_like(Wf)
    dWg=np.zeros_like(Wg); dWo=np.zeros_like(Wo)
    dUi=np.zeros_like(Ui); dUf=np.zeros_like(Uf)
    dUg=np.zeros_like(Ug); dUo=np.zeros_like(Uo)
    dbi=np.zeros_like(bi); dbf=np.zeros_like(bf)
    dbg=np.zeros_like(bg); dbo=np.zeros_like(bo)
    dh = dh_last; dc = np.zeros(h)
    dx_seq = np.zeros(T)

    for t in range(T - 1, -1, -1):
        x = x_in[t:t+1] if x_in.ndim == 1 else x_in[t]
        tanh_c = np.tanh(C[t+1])
        do_raw = dh * tanh_c * sigmoid_grad(O[t])
        dc += dh * O[t] * tanh_grad(tanh_c)
        di_raw = dc * G[t] * sigmoid_grad(I[t])
        df_raw = dc * C[t] * sigmoid_grad(F[t])
        dg_raw = dc * I[t] * tanh_grad(G[t])
        dc = dc * F[t]

        dWi += np.outer(di_raw, x); dUi += np.outer(di_raw, H[t]); dbi += di_raw
        dWf += np.outer(df_raw, x); dUf += np.outer(df_raw, H[t]); dbf += df_raw
        dWg += np.outer(dg_raw, x); dUg += np.outer(dg_raw, H[t]); dbg += dg_raw
        dWo += np.outer(do_raw, x); dUo += np.outer(do_raw, H[t]); dbo += do_raw

        dh = Ui.T@di_raw + Uf.T@df_raw + Ug.T@dg_raw + Uo.T@do_raw
        dx = Wi.T@di_raw + Wf.T@df_raw + Wg.T@dg_raw + Wo.T@do_raw
        dx_seq[t] = dx[0]   # conv output is always 1-feature per step

    return [dWi,dWf,dWg,dWo, dUi,dUf,dUg,dUo, dbi,dbf,dbg,dbo], dx_seq


def forward(x_seq: np.ndarray, weights: list) -> float:
    Wc, bc = weights[0], weights[1]
    conv_out, _ = _conv1d_fwd(x_seq.astype(np.float64), Wc, bc)
    h = HIDDEN_SIZE
    seq = conv_out
    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack_lstm(weights, i)
        H,C,I,F,G,O = _lstm_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        seq = np.array(H[1:])
    Wy, by = weights[-2], weights[-1]
    return float(np.dot(Wy, H[-1]) + by[0])


def forward_and_grad(x_seq: np.ndarray, y_true: float, weights: list):
    h = HIDDEN_SIZE
    Wc, bc = weights[0], weights[1]
    conv_out, padded = _conv1d_fwd(x_seq.astype(np.float64), Wc, bc)

    layers = []
    seq    = conv_out
    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack_lstm(weights, i)
        H,C,I,F,G,O = _lstm_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        layers.append((seq, H, C, I, F, G, O))
        seq = np.array(H[1:])

    Wy, by = weights[-2], weights[-1]
    y_pred = float(np.dot(Wy, H[-1]) + by[0])
    loss = (y_pred - y_true) ** 2
    d_pred = 2.0 * (y_pred - y_true)

    dWy = d_pred * H[-1]
    dby = np.array([d_pred])
    dh  = d_pred * Wy

    lstm_grads = []
    d_conv_out = None  #gradient into conv layer
    for i in range(NUM_LAYERS - 1, -1, -1):
        x_in,H,C,I,F,G,O = layers[i]
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack_lstm(weights, i)
        lg, dx = _lstm_bwd(x_in, H, C, I, F, G, O,
                           Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo, bi,bf,bg,bo, dh, h)
        lstm_grads.insert(0, lg)
        dh = dx if i > 0 else None
        if i == 0:
            d_conv_out = dx   #gradient into conv layer

    dWc, dbc, _ = _conv1d_bwd(d_conv_out, padded, Wc)

    flat = [dWc, dbc]
    for lg in lstm_grads: flat.extend(lg)
    flat += [dWy, dby]
    return loss, flat


def run(X: np.ndarray, y: np.ndarray, seed_seq: np.ndarray) -> dict:
    weights = init_weights()
    losses  = train(forward_and_grad, weights, X, y, NAME)
    preds   = predict_year(forward, weights, seed_seq)
    return {"name": NAME, "losses": losses, "preds_2025": preds}

# --------------------|CONVENIENCE WRAPPERS|-----------------------------------------
from train import train as _train, predict_year as _predict_year

def train_model(weights: list, X, y) -> list:
    #train in-place and return losses list
    return _train(forward_and_grad, weights, X, y, NAME)

def predict_year_from_weights(weights: list, seed_seq) -> list:
    #run autoregressive inference for PREDICT_YEAR
    return _predict_year(forward, weights, seed_seq)