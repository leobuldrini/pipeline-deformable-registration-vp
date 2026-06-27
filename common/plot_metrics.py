import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────
csv_file = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: file '{csv_file}' not found.")
    print(f"Usage: python {sys.argv[0]} <path/to/file.csv>")
    sys.exit(1)

# 7ª coluna do CSV → STSR
stsr_col = df.columns[6]

has_dsc = "val_dsc" in df.columns and df["val_dsc"].notna().any()

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#2a2d3a",
    "axes.labelcolor":  "#c8ccd8",
    "axes.grid":        True,
    "grid.color":       "#1e2130",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.8,
    "xtick.color":      "#6b7080",
    "ytick.color":      "#6b7080",
    "text.color":       "#c8ccd8",
    "font.family":      "monospace",
})

COLORS = {
    "loss": "#7dd3fc",   # sky blue
    "sim":  "#86efac",   # green
    "stsr": "#c084fc",   # purple  ← substitui reg (pink)
    "dsc":  "#fbbf24",   # amber
}
MARKERS = {"loss": "o", "sim": "s", "stsr": "^", "dsc": "D"}

# ── Figure layout ─────────────────────────────────────────────────────────────
n_plots = 4 if has_dsc else 3
fig, axes = plt.subplots(n_plots, 1, figsize=(9, 3 * n_plots + 1), sharex=True)
fig.subplots_adjust(hspace=0.08, top=0.88, bottom=0.07, left=0.10, right=0.97)

metrics = ["loss", "sim", stsr_col]
ylabels = ["Total Loss", "Similarity Loss", "STSR"]
keys    = ["loss", "sim", "stsr"]          # chaves para COLORS / MARKERS

if has_dsc:
    metrics.append("val_dsc")
    ylabels.append("Validation DSC")
    keys.append("dsc")

for ax, metric, ylabel, key in zip(axes, metrics, ylabels, keys):
    color  = COLORS[key]
    marker = MARKERS[key]

    ax.plot(
        df["epoch"], df[metric],
        color=color, linewidth=2.2,
        marker=marker, markersize=2,
        markerfacecolor=color, markeredgewidth=0.5,
        zorder=3,
    )
    ax.fill_between(df["epoch"], df[metric], alpha=0.12, color=color)

    idx_min = df[metric].idxmin()
    idx_max = df[metric].idxmax()
    for idx, va in [(idx_min, "top"), (idx_max, "bottom")]:
        ax.annotate(
            f"{df[metric][idx]:.5f}",
            xy=(df["epoch"][idx], df[metric][idx]),
            xytext=(0, 8 if va == "bottom" else -8),
            textcoords="offset points",
            ha="center", va=va,
            fontsize=7.5, color=color, alpha=0.85,
        )

    ax.set_ylabel(ylabel, fontsize=9, labelpad=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=8)

    ax.text(
        1.01, 0.5, key.upper(),
        transform=ax.transAxes,
        fontsize=8, color=color, va="center", ha="left",
        fontweight="bold",
    )

axes[-1].set_xlabel("Epoch", fontsize=9, labelpad=8)
axes[-1].set_xticks(df["epoch"])

# ── Title ─────────────────────────────────────────────────────────────────────
subtitle_parts = "Loss  ·  Similarity  ·  STSR"
if has_dsc:
    subtitle_parts += "  ·  DSC"

fig.text(0.5, 0.93, "Training Metrics per Epoch",
         ha="center", va="center", fontsize=15, fontweight="bold", color="#e2e6f0")
fig.text(0.5, 0.905, subtitle_parts,
         ha="center", va="center", fontsize=9, color="#5a6070")

output_file = csv_file.rsplit(".", 1)[0] + "_plot.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to {output_file}")
