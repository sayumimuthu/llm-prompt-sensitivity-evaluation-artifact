"""

Usage:
    python study/plot.py \\
        --in-instance study/output/combined_final/metrics_instance.csv \\
        --in-dataset  study/output/combined_final/metrics_dataset.csv \\
        --out-dir     study/output/combined_final/figures
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


matplotlib.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            9,
    "axes.titlesize":       9,
    "axes.titleweight":     "bold",
    "axes.titlepad":        5,
    "axes.labelsize":       8,
    "axes.labelpad":        3,
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "legend.fontsize":      8.5,
    "legend.title_fontsize":8.5,
    "figure.dpi":           300,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.06,
    "axes.linewidth":       0.6,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.axisbelow":       True,
    # Solid hairline grid — per dataviz spec
    "grid.linewidth":       0.5,
    "grid.color":           "#e1e0d9",
    "grid.alpha":           1.0,
    "grid.linestyle":       "-",
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
})


ZONE_COLOR = {
    "artifact":      "#eb6834",   # slot 8 — orange
    "genuine":       "#2a78d6",   # slot 1 — blue
    "underdetected": "#4a3aa7",   # slot 5 — violet
    "stable":        "#1baf7a",   # slot 2 — aqua/green
}
ZONE_LABEL = {
    "artifact":      "Artifact",
    "genuine":       "Genuine",
    "underdetected": "Underdetected",
    "stable":        "Stable",
}
ZONES = ["artifact", "genuine", "underdetected", "stable"]

# Signed-EAS and heatmap diverging poles (blue ↔ orange, neutral gray mid)
CLR_POS = "#eb6834"   # positive signed EAS: heuristic inflates
CLR_NEG = "#2a78d6"   # negative signed EAS: judge detects more
CLR_MID = "#f0efec"   # neutral midpoint

# Multi-model series colors (slots 1-8 in fixed CVD-safe order)
MODEL_COLORS = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
    "#52514e",  # secondary ink (9th model)
]

# Model display names 
MODEL_DISPLAY = {
    "gemma2:2b":          "Gemma2-2B",
    "gpt-4o":             "GPT-4o",
    "llama3.1:8b":        "Llama3.1-8B",
    "llama3.2:1b":        "Llama3.2-1B",
    "llama3.2:3b":        "Llama3.2-3B",
    "mistral-nemo:latest":"MistralNeMo-12B",
    "mistral:instruct":   "Mistral-7B",
    "qwen2.5:3b":         "Qwen2.5-3B",
    "qwen2.5:7b":         "Qwen2.5-7B",
}

DATASET_LABEL = {
    "arc_challenge": "ARC-Challenge\n(MCQ)",
    "boolq":         "BoolQ\n(Boolean)",
    "squad":         "SQuAD\n(Open-ended)",
}
DATASET_SHORT = {
    "arc_challenge": "ARC",
    "boolq":         "BoolQ",
    "squad":         "SQuAD",
}



def _name(model: str) -> str:
    """Clean display name for a model string."""
    return MODEL_DISPLAY.get(model, model.split("/")[-1])


def _style_ax(ax, ylabel: str = "", ylim=None, grid: bool = True) -> None:
    """Shared publication axis styling."""
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", length=3, width=0.6, pad=2.5)
    if grid:
        ax.grid(axis="y")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _grid_axes(n: int, per_row: int = 5,
               cell_w: float = 2.7, cell_h: float = 3.0,
               sharey: bool = True, center: bool = False):
    """Return (fig, list_of_axes[:n]).

    With center=True and a non-full last row, the last row is horizontally
    centered under the full rows using a doubled-column GridSpec so that
    the incomplete row sits in the middle rather than pinned to the left.
    """
    ncols  = min(n, per_row)
    nrows  = (n + ncols - 1) // ncols
    n_last = n - (nrows - 1) * ncols  # panels in the final row

    if center and nrows > 1 and n_last < ncols:
        # Each panel occupies 2 GridSpec columns; the last row gets
        # (ncols - n_last) empty columns as margin on each side.
        gs_cols = 2 * ncols
        fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows))
        gs  = GridSpec(nrows, gs_cols, figure=fig,
                       hspace=0.50, wspace=0.38,
                       top=0.92, bottom=0.08, left=0.08, right=0.98)
        axes = []
        for row in range(nrows - 1):                      # full rows
            for col in range(ncols):
                axes.append(fig.add_subplot(gs[row, 2*col : 2*col+2]))
        margin = ncols - n_last                           # symmetric padding
        for i in range(n_last):                           # centred last row
            c0 = margin + 2 * i
            axes.append(fig.add_subplot(gs[nrows - 1, c0 : c0 + 2]))
        if sharey:
            ref = axes[0]
            for ax in axes[1:]:
                ax.sharey(ref)
        return fig, axes

    # Plain subplots path — hide unused trailing slots
    fig, arr = plt.subplots(
        nrows, ncols,
        figsize=(cell_w * ncols, cell_h * nrows),
        sharey=sharey, squeeze=False,
    )
    flat = arr.flatten()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, list(flat[:n])


def _zone_legend_handles(zones=None):
    zones = zones or ZONES
    return [
        mpatches.Patch(facecolor=ZONE_COLOR[z], edgecolor="none",
                       label=ZONE_LABEL[z])
        for z in zones
    ]


# Figure 1: Four-zone distribution 

def figure1_trizone(inst_df: pd.DataFrame, out_path: Path) -> None:
    models   = sorted(inst_df["model_name"].unique())
    datasets = list(inst_df["dataset"].unique())

    fig, axes = _grid_axes(len(models), per_row=5, cell_w=2.7, cell_h=3.0,
                           center=True)

    x = np.arange(len(datasets))
    n_z   = len(ZONES)
    width = 0.16
    # Centre the bar cluster on each x tick
    gap = 0.015
    total = n_z * width + (n_z - 1) * gap
    offsets = np.linspace(-total / 2 + width / 2, total / 2 - width / 2, n_z)

    ncols = 5
    for i, (ax, model) in enumerate(zip(axes, models)):
        mdf = inst_df[inst_df["model_name"] == model]
        for zone, offset in zip(ZONES, offsets):
            pcts = []
            for ds in datasets:
                sub = mdf[mdf["dataset"] == ds]
                pcts.append((sub["trizone"] == zone).mean() * 100 if len(sub) else 0.0)
            ax.bar(x + offset, pcts, width,
                   color=ZONE_COLOR[zone],
                   edgecolor="white", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=7)
        ax.set_title(_name(model))
        # y-axis label only on the leftmost panel of each row
        ylabel = "% of instances" if i % ncols == 0 else ""
        _style_ax(ax, ylabel=ylabel, ylim=(0, 100))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
        if i % ncols != 0:
            ax.tick_params(labelleft=False)

    handles = _zone_legend_handles()
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), fontsize=8.5, frameon=False,
               columnspacing=1.2, handlelength=1.2, handleheight=0.85)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


# Figure 2: Scatter SensH vs SensJ 

def figure2_scatter(inst_df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(inst_df["model_name"].unique())

    fig, axes = _grid_axes(len(models), per_row=5, cell_w=2.7, cell_h=2.8, sharey=False)

    sh75 = inst_df["sens_heuristic"].quantile(0.75)
    sj75 = inst_df["sens_judge"].quantile(0.75)

    for ax, model in zip(axes, models):
        mdf = inst_df[inst_df["model_name"] == model]
        for zone in ZONES:
            sub = mdf[mdf["trizone"] == zone]
            ax.scatter(sub["sens_heuristic"], sub["sens_judge"],
                       c=ZONE_COLOR[zone], label=ZONE_LABEL[zone],
                       alpha=0.60, edgecolors="white", linewidths=0.3,
                       s=16, zorder=3, rasterized=True)

        lim = max(mdf["sens_heuristic"].max(), mdf["sens_judge"].max()) * 1.08 + 0.01
        # Identity line
        ax.plot([0, lim], [0, lim], color="#c3c2b7", linewidth=0.8,
                linestyle="--", zorder=2, label="Equal")
        # 75th-pct threshold lines (zone boundaries)
        ax.axhline(sj75, color="#898781", linewidth=0.5, linestyle=":", zorder=1)
        ax.axvline(sh75, color="#898781", linewidth=0.5, linestyle=":", zorder=1)

        ax.set_xlabel("SensH  (σ exact / F1)", fontsize=7.5)
        ax.set_ylabel("SensJ  (σ judge)", fontsize=7.5)
        ax.set_title(_name(model))
        _style_ax(ax, grid=False)
        ax.tick_params(labelsize=7)

    handles = (_zone_legend_handles()
               + [plt.Line2D([0], [0], color="#c3c2b7", linestyle="--",
                             linewidth=0.9, label="Equal")])
    fig.legend(handles=handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.01), fontsize=8.5, frameon=False,
               columnspacing=1.0, handlelength=1.2)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.8, w_pad=0.5)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


# Figure 3: Structural ablation heatmap 
def figure3_ablation(inst_df: pd.DataFrame, out_path: Path) -> None:
    models        = sorted(inst_df["model_name"].unique())
    factors       = ["role", "fmt", "prefix"]
    factor_labels = ["Role", "Format", "Prefix"]
    col_labels    = ["Heuristic", "Judge", "H − J"]
    vmax          = 0.15

    # Blue–gray–orange diverging colormap 
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pub_div", [(0.0, CLR_NEG), (0.5, CLR_MID), (1.0, CLR_POS)]
    )

    n      = len(models)
    ncols  = 5
    nrows  = (n + ncols - 1) // ncols  # 2
    n_last = n - (nrows - 1) * ncols   # 4
    margin = ncols - n_last             # 1 — symmetric padding for last row

    # 2 GridSpec columns per panel + 1 narrow colorbar column on the right.
    # This allows the bottom row to be centred: the 4 panels sit in the
    # middle of the 10-column panel space with 1 empty column on each side.
    gs_cols = 2 * ncols                 # 10 panel columns
    fig = plt.figure(figsize=(3.1 * ncols + 0.7, 3.0 * nrows))
    gs  = GridSpec(nrows, gs_cols + 1, figure=fig,
                   width_ratios=[1.0] * gs_cols + [0.11],
                   hspace=0.65, wspace=0.50,
                   top=0.95, bottom=0.06, left=0.06, right=0.96)

    model_axes = []
    for idx in range(n):
        row, col = divmod(idx, ncols)
        if row < nrows - 1:                        # full rows
            c0 = 2 * col
        else:                                      # centred last row
            c0 = margin + 2 * col
        model_axes.append(fig.add_subplot(gs[row, c0 : c0 + 2]))

    cbar_ax = fig.add_subplot(gs[:, gs_cols])  # colorbar spans all rows

    im = None
    for ax, model in zip(model_axes, models):
        mdf  = inst_df[inst_df["model_name"] == model]
        data = np.zeros((len(factors), 3))
        for fi, f in enumerate(factors):
            eh = float(mdf[f"effect_{f}_heuristic"].mean())
            ej = float(mdf[f"effect_{f}_judge"].mean())
            data[fi] = [eh, ej, eh - ej]

        im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        is_leftmost = (idx % ncols == 0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(factor_labels if is_leftmost else [], fontsize=9)
        ax.set_title(_name(model), fontsize=10, fontweight="bold", pad=5)
        ax.tick_params(left=False, bottom=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

        for fi in range(len(factors)):
            for ei in range(3):
                val   = data[fi, ei]
                color = "white" if abs(val) / vmax > 0.55 else "#0b0b0b"
                ax.text(ei, fi, f"{val:+.3f}",
                        ha="center", va="center",
                        fontsize=12, color=color, fontweight="bold")

    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Effect  (ON − OFF)", fontsize=9, labelpad=5)
        cbar.ax.tick_params(labelsize=8.5, width=0.6, length=3)
        cbar.outline.set_linewidth(0.5)
        cbar_ax.text(0.5, 1.03, "H inflates", transform=cbar_ax.transAxes,
                     ha="center", va="bottom", fontsize=8, color="#52514e")
        cbar_ax.text(0.5, -0.03, "J inflates", transform=cbar_ax.transAxes,
                     ha="center", va="top",    fontsize=8, color="#52514e")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Figure 4: EAS by task type 

def figure4_eas_by_task(inst_df: pd.DataFrame, out_path: Path) -> None:
    models  = sorted(inst_df["model_name"].unique())
    tasks   = ["mcq", "boolean", "open_ended"]
    t_label = {"mcq": "MCQ", "boolean": "Boolean", "open_ended": "Open-ended"}

    fig, axes = _grid_axes(len(models), per_row=5, cell_w=2.7, cell_h=3.0)

    # Task colors: genuine=MCQ, stable=Boolean, artifact=Open-ended
    task_colors = [ZONE_COLOR["genuine"], ZONE_COLOR["stable"], ZONE_COLOR["artifact"]]

    for ax, model in zip(axes, models):
        mdf  = inst_df[inst_df["model_name"] == model]
        data = [mdf.loc[mdf["task_type"] == t, "eas"].dropna().values for t in tasks]
        bp   = ax.boxplot(data, patch_artist=True, widths=0.48,
                          medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                          whiskerprops={"linewidth": 0.7, "color": "#52514e"},
                          capprops=  {"linewidth": 0.7, "color": "#52514e"},
                          boxprops=  {"linewidth": 0.6},
                          flierprops={"marker": "o", "markersize": 2.0,
                                      "markerfacecolor": "#898781",
                                      "markeredgewidth": 0.0,
                                      "linestyle": "none", "alpha": 0.6})
        for patch, color in zip(bp["boxes"], task_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor("#0b0b0b")

        ax.set_xticks(range(1, len(tasks) + 1))
        ax.set_xticklabels([t_label[t] for t in tasks], fontsize=7.5)
        ax.set_title(_name(model))
        _style_ax(ax, ylabel="EAS")

    plt.tight_layout(h_pad=0.8, w_pad=0.5)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


# Figure 5: Mean EAS with bootstrap CI 

def figure5_eas_ci(agg_df: pd.DataFrame, out_path: Path) -> None:
    models   = sorted(agg_df["model_name"].unique())
    datasets = list(agg_df["dataset"].unique())
    has_ci   = "eas_ci_lo" in agg_df.columns

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(models)), 3.8))
    x     = np.arange(len(datasets))
    width = 0.65 / len(models)

    for mi, model in enumerate(models):
        mdf                     = agg_df[agg_df["model_name"] == model]
        means, lo_errs, hi_errs = [], [], []
        for ds in datasets:
            row = mdf[mdf["dataset"] == ds]
            if not len(row):
                means.append(0); lo_errs.append(0); hi_errs.append(0)
                continue
            mean = float(row["eas_mean"].iloc[0])
            means.append(mean)
            if has_ci:
                lo_errs.append(max(0.0, mean - float(row["eas_ci_lo"].iloc[0])))
                hi_errs.append(max(0.0, float(row["eas_ci_hi"].iloc[0]) - mean))
            else:
                lo_errs.append(0); hi_errs.append(0)

        offset = (mi - (len(models) - 1) / 2) * (width + 0.01)
        kw: dict = dict(
            label=_name(model),
            color=MODEL_COLORS[mi % len(MODEL_COLORS)],
            edgecolor="white", linewidth=0.6,
        )
        if has_ci:
            kw["yerr"]     = [lo_errs, hi_errs]
            kw["capsize"]  = 2.5
            kw["error_kw"] = {"linewidth": 0.9, "ecolor": "#52514e", "capthick": 0.9}
        ax.bar(x + offset, means, width, **kw)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=8.5)
    _style_ax(ax, ylabel="Mean EAS")
    ax.legend(fontsize=7.5, frameon=False, ncol=3,
              loc="upper left", handlelength=1.0, columnspacing=0.8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")

# Figure 6: Signed EAS 

def figure6_signed_eas(inst_df: pd.DataFrame, out_path: Path) -> None:
    if "signed_eas" not in inst_df.columns:
        print(f"  Skipping {out_path.name} — signed_eas column missing (re-run metrics.py)")
        return

    models   = sorted(inst_df["model_name"].unique())
    datasets = list(inst_df["dataset"].unique())

    fig, axes = _grid_axes(len(models), per_row=5, cell_w=2.7, cell_h=3.0,
                           center=True)

    ncols = 5
    for i, (ax, model) in enumerate(zip(axes, models)):
        mdf   = inst_df[inst_df["model_name"] == model]
        x     = np.arange(len(datasets))
        means = [float(mdf.loc[mdf["dataset"] == ds, "signed_eas"].mean())
                 for ds in datasets]
        colors = [CLR_POS if m >= 0 else CLR_NEG for m in means]

        ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="#0b0b0b", linewidth=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=7)
        ax.set_title(_name(model))
        ylabel = "Mean Signed EAS" if i % ncols == 0 else ""
        _style_ax(ax, ylabel=ylabel)
        if i % ncols != 0:
            ax.tick_params(labelleft=False)

    handles = [
        mpatches.Patch(facecolor=CLR_POS, edgecolor="none",
                       label="Heuristic inflates  (H > J)"),
        mpatches.Patch(facecolor=CLR_NEG, edgecolor="none",
                       label="Judge detects more  (J > H)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.995), fontsize=8.5, frameon=False,
               columnspacing=1.5, handlelength=1.2)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved {out_path}")


# Main 
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures.")
    parser.add_argument("--in-instance", default="study/output/combined_final/metrics_instance.csv")
    parser.add_argument("--in-dataset",  default="study/output/combined_final/metrics_dataset.csv")
    parser.add_argument("--out-dir",     default="study/output/combined_final/figures")
    args = parser.parse_args()

    inst_df = pd.read_csv(args.in_instance)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        agg_df = pd.read_csv(args.in_dataset)
    except FileNotFoundError:
        agg_df = None
        print("  Warning: dataset CSV not found — skipping Fig 5")

    print("Generating figures...")
    figure1_trizone(inst_df,     out_dir / "fig1_trizone.png")
    figure2_scatter(inst_df,     out_dir / "fig2_scatter.png")
    figure3_ablation(inst_df,    out_dir / "fig3_ablation.png")
    figure4_eas_by_task(inst_df, out_dir / "fig4_eas_by_task.png")
    if agg_df is not None:
        figure5_eas_ci(agg_df,   out_dir / "fig5_eas_ci.png")
    figure6_signed_eas(inst_df,  out_dir / "fig6_signed_eas.png")
    print("Done.")


if __name__ == "__main__":
    main()
