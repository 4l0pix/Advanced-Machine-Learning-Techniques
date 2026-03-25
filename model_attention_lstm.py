# --------------------------------------------------------------
#  model_attention_lstm.py  –  LSTM + Bahdanau Attention (BPTT)
#
#  All hidden states from the final LSTM layer are retained.
#  Additive attention computes a weighted context vector which
#  is projected to the scalar prediction.
# --------------------------------------------------------------

import numpy as np
from config import HIDDEN_SIZE, NUM_LAYERS, SEED
from train import train, predict_year

NAME = "LSTM + Attention"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def sigmoid_grad(s): return s * (1.0 - s)
def tanh_grad(t): return 1.0 - t * t


def init_weights() -> list:
    """
    LSTM layers: NUM_LAYERS × 12 arrays
    Attention:   Wa (h,h)  va (h,)
    Output:      Wy (h,)   by (1,)
    """
    rng = np.random.default_rng(SEED)
    h = HIDDEN_SIZE
    ws = []
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
    sa = np.sqrt(1.0 / h)
    ws += [rng.normal(0, sa, (h, h)),   #wa
           rng.normal(0, sa, h),         #va
           rng.normal(0, 0.1, h),        #wy
           np.zeros(1)]   #by
    return [w.astype(np.float64) for w in ws]


def _unpack_lstm(weights, layer_idx):
    base = layer_idx * 12
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


def _attention_fwd(H_mat, Wa, va):
    """
    H_mat: (T, h)  — all hidden states from last LSTM layer
    Returns context (h,), alpha (T,), tanh_scores (T, h)
    """
    tanh_scores = np.tanh(H_mat @ Wa.T)   #(T, h)
    scores = tanh_scores @ va   #(T,)
    scores -= scores.max()
    alpha = np.exp(scores)
    alpha /= alpha.sum()
    context = alpha @ H_mat   #(h,)
    return context, alpha, tanh_scores


def forward(x_seq: np.ndarray, weights: list) -> float:
    h = HIDDEN_SIZE
    seq = x_seq.astype(np.float64)
    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack_lstm(weights, i)
        H,C,I,F,G,O = _lstm_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        seq = np.array(H[1:])

    attn_end = NUM_LAYERS * 12
    Wa, va, Wy, by = weights[attn_end], weights[attn_end+1], weights[attn_end+2], weights[attn_end+3]
    H_mat = np.array(H[1:])   #(T, h)
    context, _, _ = _attention_fwd(H_mat, Wa, va)
    return float(np.dot(Wy, context) + by[0])


def forward_and_grad(x_seq: np.ndarray, y_true: float, weights: list):
    h = HIDDEN_SIZE
    T = len(x_seq)
    layers = []
    seq = x_seq.astype(np.float64)

    for i in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack_lstm(weights, i)
        H,C,I,F,G,O = _lstm_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        layers.append((seq, H, C, I, F, G, O))
        seq = np.array(H[1:])

    attn_end = NUM_LAYERS * 12
    Wa = weights[attn_end];   va = weights[attn_end+1]
    Wy = weights[attn_end+2]; by = weights[attn_end+3]

    H_mat = np.array(H[1:])   #(T, h)
    context, alpha, tanh_scores = _attention_fwd(H_mat, Wa, va)

    y_pred = float(np.dot(Wy, context) + by[0])
    loss = (y_pred - y_true) ** 2
    d_pred = 2.0 * (y_pred - y_true)

    dWy = d_pred * context
    dby = np.array([d_pred])
    d_context = d_pred * Wy   #(h,)

    # --------------------|ATTENTION BACKWARD|---------------------------------
    #context = alpha @ H_mat  →  d_H_mat_attn and d_alpha
    d_H_attn = alpha[:, None] * d_context[None, :]   #(T, h)
    d_alpha = H_mat @ d_context   #(T,)

    #alpha = softmax(scores)  →  d_scores
    d_scores = alpha * (d_alpha - np.dot(alpha, d_alpha))   #(T,)

    #scores = tanh_scores @ va  →  d_va, d_tanh_scores
    dva = tanh_scores.T @ d_scores   #(h,)
    d_tanh_scores = np.outer(d_scores, va)   #(T, h)

    #tanh_scores = tanh(H_mat @ Wa.T)  →  dWa, d_H_mat_wa
    d_inner = d_tanh_scores * (1.0 - tanh_scores ** 2)   #(T, h)
    dWa = d_inner.T @ H_mat   #(h, h)
    d_H_wa = d_inner @ Wa   #(T, h)  ← grad through Wa

    #total gradient into H_mat from attention
    d_H_mat = d_H_attn + d_H_wa   #(T, h)

    # |LSTM BPTT|----------------------------------------------------
    #d_H_mat gives gradient at each timestep's hidden state
    all_grads = []
    dh_next = np.zeros(h)   #no gradient from beyond last layer

    for i in range(NUM_LAYERS - 1, -1, -1):
        x_in, H, C, I, F, G, O = layers[i]
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack_lstm(weights, i)

        dWi=np.zeros_like(Wi); dWf=np.zeros_like(Wf)
        dWg=np.zeros_like(Wg); dWo=np.zeros_like(Wo)
        dUi=np.zeros_like(Ui); dUf=np.zeros_like(Uf)
        dUg=np.zeros_like(Ug); dUo=np.zeros_like(Uo)
        dbi=np.zeros_like(bi); dbf=np.zeros_like(bf)
        dbg=np.zeros_like(bg); dbo=np.zeros_like(bo)
        dx_seq = np.zeros_like(x_in)

        dh = dh_next
        dc = np.zeros(h)

        for t in range(T - 1, -1, -1):
            x = x_in[t] if x_in.ndim > 1 else x_in[t:t+1]
            tanh_c = np.tanh(C[t+1])

            #add attention gradient at this timestep (only for last LSTM layer)
            if i == NUM_LAYERS - 1:
                dh = dh + d_H_mat[t]

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
            dx_seq[t] = dx if x_in.ndim > 1 else dx[0]

        all_grads.insert(0, [dWi,dWf,dWg,dWo, dUi,dUf,dUg,dUo, dbi,dbf,dbg,dbo])
        dh_next = dh

    flat = []
    for lg in all_grads: flat.extend(lg)
    flat += [dWa, dva, dWy, dby]
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