# --------------------------------------------------------------
#  model_lstm.py  –  Classic stacked LSTM  (BPTT)
# --------------------------------------------------------------

import numpy as np
from config import HIDDEN_SIZE, NUM_LAYERS, SEED
from train import train, predict_year

NAME = "Classic LSTM"

def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(s): return s * (1.0 - s)   #s already sigmoid(x)
def tanh_grad(t): return 1.0 - t * t   #t already tanh(x)


# --------------------------|WEIGHT HELPERS|----------------------------------

def init_weights() -> list:
    """
    Per layer (NUM_LAYERS total):
      Wi (h, in)  Wf (h, in)  Wg (h, in)  Wo (h, in)
      Ui (h, h)   Uf (h, h)   Ug (h, h)   Uo (h, h)
      bi (h,)     bf (h,)     bg (h,)      bo (h,)
    Output layer:
      Wy (h,)  by (1,)
    All arrays are float64 for numerical stability.
    """
    rng = np.random.default_rng(SEED)
    h = HIDDEN_SIZE
    ws = []
    for i in range(NUM_LAYERS):
        d = 1 if i == 0 else h
        s = np.sqrt(2.0 / (d + h))
        ws += [
            rng.normal(0, s, (h, d)),   # Wi
            rng.normal(0, s, (h, d)),   # Wf
            rng.normal(0, s, (h, d)),   # Wg
            rng.normal(0, s, (h, d)),   # Wo
            rng.normal(0, s, (h, h)),   # Ui
            rng.normal(0, s, (h, h)),   # Uf
            rng.normal(0, s, (h, h)),   # Ug
            rng.normal(0, s, (h, h)),   # Uo
            np.zeros(h),                # bi
            np.ones(h),                 # bf  (forget bias = 1)
            np.zeros(h),                # bg
            np.zeros(h),                # bo
        ]
    ws += [rng.normal(0, 0.1, h), np.zeros(1)]   # Wy, by
    return [w.astype(np.float64) for w in ws]


def _unpack(weights, layer_idx):
    base = layer_idx * 12
    return (weights[base],   weights[base+1], weights[base+2],  weights[base+3],
            weights[base+4], weights[base+5], weights[base+6],  weights[base+7],
            weights[base+8], weights[base+9], weights[base+10], weights[base+11])


# -------------------------------|FORWARD PASS|-------------------------------------

def _lstm_layer_fwd(x_seq, Wi, Wf, Wg, Wo, Ui, Uf, Ug, Uo, bi, bf, bg, bo, h):
    #run one LSTM layer. Returns lists of gates/states for BPTT.
    T = len(x_seq)
    H = [np.zeros(h)]   #h[0] = h_prev at t=0
    C = [np.zeros(h)]   #c[0] = c_prev at t=0
    I, F, G, O = [], [], [], []  #gate activations (after nonlinearity)

    for t in range(T):
        x = x_seq[t] if x_seq.ndim > 1 else x_seq[t:t+1]
        i_g = sigmoid(Wi @ x + Ui @ H[t] + bi)
        f_g = sigmoid(Wf @ x + Uf @ H[t] + bf)
        g_g = np.tanh (Wg @ x + Ug @ H[t] + bg)
        o_g = sigmoid(Wo @ x + Uo @ H[t] + bo)
        c = f_g * C[t] + i_g * g_g
        hh = o_g * np.tanh(c)
        I.append(i_g); F.append(f_g); G.append(g_g); O.append(o_g)
        H.append(hh); C.append(c)

    return H, C, I, F, G, O


def forward(x_seq: np.ndarray, weights: list) -> float:
    h = HIDDEN_SIZE
    seq = x_seq.astype(np.float64)
    for layer in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, layer)
        H,C,I,F,G,O = _lstm_layer_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        seq = np.array(H[1:])     #(T, h) — feed to next layer
    Wy, by = weights[-2], weights[-1]
    return float(np.dot(Wy, H[-1]) + by[0])


# -------------------------------|BACKWARD PASS|----------------------------------

def forward_and_grad(x_seq: np.ndarray, y_true: float, weights: list):
    h = HIDDEN_SIZE
    T = len(x_seq)
    layers = []
    seq = x_seq.astype(np.float64)

    #forward through all layers, cache everything
    for layer in range(NUM_LAYERS):
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, layer)
        H,C,I,F,G,O = _lstm_layer_fwd(seq, Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo, h)
        layers.append((seq, H, C, I, F, G, O))
        seq = np.array(H[1:])    #output sequence for next layer

    Wy, by = weights[-2], weights[-1]
    y_pred = float(np.dot(Wy, H[-1]) + by[0])
    loss = (y_pred - y_true) ** 2
    d_pred = 2.0 * (y_pred - y_true)   #dL/dy_pred

    #gradients for output layer
    dWy = d_pred * H[-1]
    dby = np.array([d_pred])
    dh_next = d_pred * Wy   #gradient flowing back into last LSTM layer

    all_grads = []

    #backprop through LSTM layers (reverse order)
    for layer in range(NUM_LAYERS - 1, -1, -1):
        x_in, H, C, I, F, G, O = layers[layer]
        Wi,Wf,Wg,Wo,Ui,Uf,Ug,Uo,bi,bf,bg,bo = _unpack(weights, layer)

        dWi=np.zeros_like(Wi); dWf=np.zeros_like(Wf)
        dWg=np.zeros_like(Wg); dWo=np.zeros_like(Wo)
        dUi=np.zeros_like(Ui); dUf=np.zeros_like(Uf)
        dUg=np.zeros_like(Ug); dUo=np.zeros_like(Uo)
        dbi=np.zeros_like(bi); dbf=np.zeros_like(bf)
        dbg=np.zeros_like(bg); dbo=np.zeros_like(bo)

        dh = dh_next
        dc = np.zeros(h)
        dx_seq = np.zeros_like(x_in)

        for t in range(T - 1, -1, -1):
            x = x_in[t] if x_in.ndim > 1 else x_in[t:t+1]
            tanh_c = np.tanh(C[t+1])

            #output gate
            do = dh * tanh_c
            do_raw = do * sigmoid_grad(O[t])
            #cell state
            dc += dh * O[t] * tanh_grad(tanh_c)
            #forget gate
            df = dc * C[t]
            df_raw = df * sigmoid_grad(F[t])
            #input gate
            di = dc * G[t]
            di_raw = di * sigmoid_grad(I[t])
            #gate (candidate)
            dg = dc * I[t]
            dg_raw = dg * tanh_grad(G[t])
            #next cell state gradient
            dc = dc * F[t]

            #accumulate weight gradients
            dWi += np.outer(di_raw, x)
            dWf += np.outer(df_raw, x)
            dWg += np.outer(dg_raw, x)
            dWo += np.outer(do_raw, x)
            dUi += np.outer(di_raw, H[t])
            dUf += np.outer(df_raw, H[t])
            dUg += np.outer(dg_raw, H[t])
            dUo += np.outer(do_raw, H[t])
            dbi += di_raw; dbf += df_raw; dbg += dg_raw; dbo += do_raw

            #gradient to previous hidden state
            dh = (Ui.T @ di_raw + Uf.T @ df_raw +
                  Ug.T @ dg_raw + Uo.T @ do_raw)

            #gradient to input (for stacked layers)
            dx = (Wi.T @ di_raw + Wf.T @ df_raw +
                  Wg.T @ dg_raw + Wo.T @ do_raw)
            if x_in.ndim > 1:
                dx_seq[t] = dx
            else:
                dx_seq[t] = dx[0]

        dh_next = dh   #pass gradient to layer below

        #store in reverse so we rebuild in original order at the end
        all_grads.insert(0, [dWi, dWf, dWg, dWo,
                              dUi, dUf, dUg, dUo,
                              dbi, dbf, dbg, dbo])

    #flatten grads in same order as weights
    flat_grads = []
    for layer_grads in all_grads:
        flat_grads.extend(layer_grads)
    flat_grads += [dWy, dby]

    return loss, flat_grads


def run(X: np.ndarray, y: np.ndarray, seed_seq: np.ndarray) -> dict:
    weights = init_weights()
    losses = train(forward_and_grad, weights, X, y, NAME)
    preds = predict_year(forward, weights, seed_seq)
    return {"name": NAME, "losses": losses, "preds_2025": preds}

# --------------|convenience wrappers used by main.py|--------------------
from train import train as _train, predict_year as _predict_year

def train_model(weights: list, X, y) -> list:
    #train in-place and return losses list
    return _train(forward_and_grad, weights, X, y, NAME)

def predict_year_from_weights(weights: list, seed_seq) -> list:
    #run autoregressive inference for PREDICT_YEAR
    return _predict_year(forward, weights, seed_seq)