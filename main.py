# --------------------------------------------------------------
# main.py  –  train LSTM family models, evaluate, and plot results
# --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

from config import TRAIN_YEARS, PREDICT_YEAR, MONTHS
from data   import load_daily, build_monthly, prepare_training_data

import model_lstm
import model_bilstm
import model_gru
import model_convlstm
import model_attention_lstm

MODELS = [
    model_lstm,
    model_bilstm,
    model_gru,
    model_convlstm,
    model_attention_lstm,
]

COLORS = {
    "Classic LSTM":     "#4C9BE8",
    "Bi-LSTM":          "#E8834C",
    "GRU":              "#4CE87A",
    "ConvLSTM":         "#C04CE8",
    "LSTM + Attention": "#E8D54C",
}

BG_DARK = "#0F1117"
BG_PANEL = "#1A1D27"
GRID_COL = "#2A2D3A"


# ----------------|METRICS|----------------

def compute_metrics(result: dict, X: np.ndarray, y: np.ndarray, mu: float, sigma: float, model_module) -> dict:
    #evaluate the trained model on the training sequences.
    
    preds_norm = np.array([model_module.forward(x, result["weights"]) for x in X])
    y_real = y * sigma + mu
    p_real = preds_norm * sigma + mu

    residuals = y_real - p_real
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    final_loss = result["losses"][-1]
    #returns a dict with MSE, RMSE (°C), MAE (°C), R^2, final_loss.
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "final_loss": final_loss}


# ----------------|HELPER(just for the aesthetics:)|----------------

def _style_ax(ax, title=None, xlabel=None, ylabel=None, fontsize=11):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color(GRID_COL)
    ax.grid(alpha=0.18, color="white", linestyle="--", linewidth=0.5)
    if title:   ax.set_title(title,   color="white", fontsize=fontsize, pad=8)
    if xlabel:  ax.set_xlabel(xlabel, color="white", fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, color="white", fontsize=9)


# ----------------|PLOTING|----------------

def plot_results(results: list, metrics: list,
                 monthly: pd.DataFrame, mu: float, sigma: float):
    """
    Layout (5 rows × 2 cols):
      Row 0  [full-width]  – 2025 predictions
      Row 1  [full-width]  – training loss curves (annotated)
      Row 2  left          – RMSE (°C) bar chart per model
             right         – MAE  (°C) bar chart per model
      Row 3  left          – R^2        bar chart per model
             right         – model disagreement (spread)
      Row 4  [full-width]  – metrics summary table
    """
    fig = plt.figure(figsize=(22, 24), facecolor=BG_DARK)
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.32, top=0.95, bottom=0.03)

    ax_pred = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1, :])
    ax_rmse = fig.add_subplot(gs[2, 0])
    ax_mae = fig.add_subplot(gs[2, 1])
    ax_r2 = fig.add_subplot(gs[3, 0])
    ax_spread = fig.add_subplot(gs[3, 1])
    ax_table = fig.add_subplot(gs[4, :])

    names = [r["name"] for r in results]
    colors = [COLORS[n]  for n in names]

    # ----------------Row 0: 2025 predictions----------------------
    actual = monthly[monthly["year"] == PREDICT_YEAR]
    if not actual.empty:
        ax_pred.plot(range(1, len(actual) + 1), actual["temp"].values, "w--", linewidth=2.5, label="Actual 2025", zorder=5)

    all_preds = []
    for r in results:
        preds_real = np.array(r["preds_2025"]) * sigma + mu
        all_preds.append(preds_real)
        ax_pred.plot(range(1, 13), preds_real, color=COLORS[r["name"]], linewidth=2, marker="o", markersize=6, label=r["name"])

    _style_ax(ax_pred, title="2025 Monthly Average Temperature Predictions – Karditsa, Greece", ylabel="Temperature (°C)", fontsize=13)
    ax_pred.set_xticks(range(1, 13))
    ax_pred.set_xticklabels(MONTHS, color="white")
    ax_pred.legend(facecolor=BG_PANEL, labelcolor="white", fontsize=9, framealpha=0.8, edgecolor=GRID_COL)

    # ------------------Row 1: training loss + final-loss annotations -------------------------
    for r, m in zip(results, metrics):
        line, = ax_loss.plot(r["losses"], color=COLORS[r["name"]], linewidth=1.5, label=r["name"])
        #annotate final loss at the right end
        final_epoch = len(r["losses"]) - 1
        final_val = r["losses"][-1]
        ax_loss.annotate(
            f'{final_val:.4f}',
            xy=(final_epoch, final_val),
            xytext=(final_epoch - len(r["losses"]) * 0.08, final_val * 1.6),
            color=COLORS[r["name"]], fontsize=7.5, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLORS[r["name"]],
                            lw=0.8, alpha=0.6),
        )

    _style_ax(ax_loss, title="Training Loss per Epoch  (MSE, normalised scale)", xlabel="Epoch", ylabel="MSE (log scale)", fontsize=13)
    ax_loss.set_yscale("log")
    ax_loss.legend(facecolor=BG_PANEL, labelcolor="white", fontsize=9, framealpha=0.8, edgecolor=GRID_COL)

    # ---------------------- Row 2 left: RMSE ----------------------
    rmse_vals = [m["RMSE"] for m in metrics]
    bars = ax_rmse.bar(names, rmse_vals, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, rmse_vals):
        ax_rmse.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}°C', ha="center", va="bottom",
                     color="white", fontsize=8, fontweight="bold")
    _style_ax(ax_rmse, title="RMSE on Training Set (°C)", ylabel="RMSE (°C)")
    ax_rmse.set_xticklabels(names, rotation=15, ha="right", color="white", fontsize=8)

    # ---------------------- Row 2 right: MAE ----------------------
    mae_vals = [m["MAE"] for m in metrics]
    bars = ax_mae.bar(names, mae_vals, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, mae_vals):
        ax_mae.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}°C', ha="center", va="bottom",
                    color="white", fontsize=8, fontweight="bold")
    _style_ax(ax_mae, title="MAE on Training Set (°C)", ylabel="MAE (°C)")
    ax_mae.set_xticklabels(names, rotation=15, ha="right", color="white", fontsize=8)

    # ---------------------- Row 3 left: R^2 ----------------------
    r2_vals = [m["R2"] for m in metrics]
    bars = ax_r2.bar(names, r2_vals, color=colors, alpha=0.85, width=0.6)
    for bar, val in zip(bars, r2_vals):
        ax_r2.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.002,
                   f'{val:.4f}', ha="center", va="bottom",
                   color="white", fontsize=8, fontweight="bold")
        
    _style_ax(ax_r2, title="R² Score on Training Set  (higher = better)", ylabel="R²")
    ax_r2.set_xticklabels(names, rotation=15, ha="right", color="white", fontsize=8)
    ax_r2.set_ylim(0, min(1.05, max(r2_vals) * 1.15))

    # ---------------------- Row 3 right: model spread ----------------------
    spread = np.std(all_preds, axis=0)
    ax_spread.bar(range(1, 13), spread, color="#E8834C", alpha=0.85)
    for i, val in enumerate(spread):
        ax_spread.text(i + 1, val + 0.01, f'{val:.2f}', ha="center", va="bottom", color="white", fontsize=7)

    _style_ax(ax_spread, title="Model Disagreement – Std Dev Across Models (°C)", ylabel="Std Dev (°C)")
    ax_spread.set_xticks(range(1, 13))
    ax_spread.set_xticklabels(MONTHS, fontsize=8, color="white")

    # ---------------------- Row 4: summary metrics table ----------------------
    ax_table.set_facecolor(BG_PANEL)
    ax_table.axis("off")
    ax_table.set_title("Model Metrics Summary", color="white", fontsize=12, pad=10, loc="left")

    col_labels = ["Model", "Final Loss (MSE)", "RMSE (°C)", "MAE(°C)", "R^2"]
    rows = []
    for r, m in zip(results, metrics):
        rows.append([
            r["name"],
            f'{m["final_loss"]:.5f}',
            f'{m["RMSE"]:.3f}',
            f'{m["MAE"]:.3f}',
            f'{m["R2"]:.4f}',
        ])

    tbl = ax_table.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    #style header row
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#2C3E6B")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    #Style data rows — highlight best value per metric column in green
    metric_keys  = ["final_loss", "RMSE", "MAE", "R2"]
    best_is_low  = [True, True, True, False]

    for col_idx, (key, low) in enumerate(zip(metric_keys, best_is_low)):
        vals = [m[key] for m in metrics]
        best = min(vals) if low else max(vals)
        for row_idx, (val, r) in enumerate(zip(vals, results)):
            cell = tbl[row_idx + 1, col_idx + 1]   # +1 for header and name col
            is_best = abs(val - best) < 1e-9
            cell.set_facecolor("#1B3A2B" if is_best else "#1E2235")
            cell.set_text_props(color="#4CE87A" if is_best else "white",fontweight="bold" if is_best else "normal")
        # Style name column
        for row_idx, r in enumerate(results):
            name_cell = tbl[row_idx + 1, 0]
            name_cell.set_facecolor("#1E2235")
            name_cell.set_text_props(color=COLORS[r["name"]], fontweight="bold")

    fig.suptitle("LSTM Family – Monthly Temperature Forecast  |  Karditsa, Greece",
                 color="white", fontsize=17, fontweight="bold")

    out = "lstm_weather_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {out}")
    plt.close()


# ----------------|MAIN|----------------

def main():
    print("=" * 60)
    print("LSTM Weather Prediction – Karditsa, Greece")
    print("=" * 60)

    # 1. load the data
    print("\n[1/3] Loading weather data")
    daily = load_daily()
    monthly = build_monthly(daily)
    X, y, norm_train, mu, sigma = prepare_training_data(monthly)
    seed_seq = norm_train

    print(f"μ = {mu:.2f} °C   σ = {sigma:.2f} °C   sequences = {len(X)}")

    # 2. train each model — store weights in result for metric computation
    print("\n[2/3] training models\n")
    results = []
    for i, model in enumerate(MODELS, 1):
        print(f"  [{i}/{len(MODELS)}] {model.NAME}")
        weights = model.init_weights()
        losses  = model.train_model(weights, X, y)
        preds   = model.predict_year_from_weights(weights, seed_seq)
        result  = {"name": model.NAME, "losses": losses, "preds_2025": preds, "weights": weights}
        results.append(result)
        print(f"Final MSE: {losses[-1]:.5f}\n")

    # 3. Compute metrics on training set
    metrics = [compute_metrics(r, X, y, mu, sigma, mod)
               for r, mod in zip(results, MODELS)]

    # 4. Print table
    print("[3/3] 2025 Predictions (°C)\n")
    names  = [r["name"] for r in results]
    header = f"{'Month':<6}" + "".join(f"{n:>20}" for n in names)
    print(header)
    print("-" * len(header))
    for i, month in enumerate(MONTHS):
        row = f"{month:<6}"
        for r in results:
            val  = r["preds_2025"][i] * sigma + mu
            row += f"{val:>20.2f}"
        print(row)

    print("\nMetrics (training set):\n")
    print(f"  {'Model':<22}  {'Final MSE':>12}  {'RMSE(°C)':>10}  {'MAE(°C)':>10}  {'R^2':>8}")
    for r, m in zip(results, metrics):
        print(f"  {r['name']:<22}  {m['final_loss']:>12.5f}  "
              f"{m['RMSE']:>10.3f}  {m['MAE']:>10.3f}  {m['R2']:>8.4f}")

    plot_results(results, metrics, monthly, mu, sigma)
    print("\ndone!")


if __name__ == "__main__":
    main()