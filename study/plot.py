"""
Generate the figures from metrics_instance.csv.

Figure 1: Tri-zone bar chart — fraction of instances in each zone, by dataset × model.
Figure 2: Scatter — SensHeuristic vs SensJudge, colored by tri-zone, per model.
Figure 3: Structural ablation heatmap — factor effects on heuristic vs judge scores.
Figure 4 : EAS by task type
Figure 5: Mean EAS with bootstrap CI by dataset × model
Figure 6: Signed EAS

Usage:
    python study/plot.py \
        --in-instance study/output/metrics_instance.csv \
        --out-dir     study/output/figures
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Colour palette

ZONE_COLOR = {"artifact": "#E07B54", "genuine": "#5480E0", "stable": "#54C278",
              "underdetected": "#C278D4"}
ZONE_LABEL = {"artifact": "Artifact",  "genuine": "Genuine",  "stable": "Stable",
              "underdetected": "Underdetected"}
ZONES      = ["artifact", "genuine", "underdetected", "stable"]

DATASET_LABEL = {
    "arc_challenge": "ARC-Challenge\n(MCQ)",
    "boolq":         "BoolQ\n(Boolean)",
    "squad":         "SQuAD\n(Open-ended)",
}


# Grid layout helper

def _grid_axes(n: int, per_row: int = 4,
               cell_w: float = 5.0, cell_h: float = 4.5,
               sharey: bool = True):
    """Return (fig, axes_flat) with at most per_row panels per row.

    Unused axes in the last row are hidden automatically.
    """
    ncols = min(n, per_row)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell_w * ncols, cell_h * nrows),
        sharey=sharey,
        squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    return fig, axes_flat[:n]


# Figure 1: Tri-zone bar chart


def figure1_trizone(inst_df: pd.DataFrame, out_path: Path) -> None:
    models   = sorted(inst_df["model_name"].unique())
    datasets = list(inst_df["dataset"].unique())

    fig, axes = _grid_axes(len(models), cell_w=5.0, cell_h=4.5)

    x     = np.arange(len(datasets))
    width = 0.18

    for ax, model in zip(axes, models):
        mdf = inst_df[inst_df["model_name"] == model]
        for zi, zone in enumerate(ZONES):
            pcts = []
            for ds in datasets:
                sub = mdf[mdf["dataset"] == ds]
                pct = (sub["trizone"] == zone).mean() * 100 if len(sub) else 0.0
                pcts.append(pct)
            offset = (zi - 1.5) * (width + 0.02)
            ax.bar(x + offset, pcts, width,
                   label=ZONE_LABEL[zone], color=ZONE_COLOR[zone],
                   alpha=0.88, edgecolor="white", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=9)
        short_model = model.split("/")[-1]
        ax.set_title(short_model, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel("% of instances", fontsize=9)
        ax.set_ylim(0, 100)
        ax.yaxis.set_tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(color=ZONE_COLOR[z], label=ZONE_LABEL[z]) for z in ZONES]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), fontsize=9, frameon=False)
    fig.suptitle("Figure 1  Four-Zone Distribution by Dataset and Model",
                 y=1.05, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Figure 2: Scatter SensHeuristic vs SensJudge


def figure2_scatter(inst_df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(inst_df["model_name"].unique())

    fig, axes = _grid_axes(len(models), cell_w=5.0, cell_h=4.5)

    for ax, model in zip(axes, models):
        mdf = inst_df[inst_df["model_name"] == model]

        for zone in ZONES:
            sub = mdf[mdf["trizone"] == zone]
            ax.scatter(sub["sens_heuristic"], sub["sens_judge"],
                       c=ZONE_COLOR[zone], label=ZONE_LABEL[zone],
                       alpha=0.70, edgecolors="white", linewidths=0.4, s=55, zorder=3)

        lim = max(mdf["sens_heuristic"].max(), mdf["sens_judge"].max()) * 1.05 + 0.01
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.9, alpha=0.35, label="Equal", zorder=2)

        ax.axhline(inst_df["sens_judge"].quantile(0.75),     color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axvline(inst_df["sens_heuristic"].quantile(0.75), color="grey", linewidth=0.5, linestyle=":", alpha=0.6)

        ax.set_xlabel("SensHeuristic  (σ exact / F1)", fontsize=9)
        ax.set_ylabel("SensJudge  (σ judge score)", fontsize=9)
        short_model = model.split("/")[-1]
        ax.set_title(short_model, fontsize=10, fontweight="bold", pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    handles = ([mpatches.Patch(color=ZONE_COLOR[z], label=ZONE_LABEL[z]) for z in ZONES]
               + [plt.Line2D([0], [0], color="k", linestyle="--", linewidth=0.9, label="Equal")])  # noqa: E501
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.01), fontsize=9, frameon=False)
    fig.suptitle("Figure 2  Heuristic vs. Judge Sensitivity per Instance",
                 y=1.05, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Figure 3: Structural ablation heatmap


def figure3_ablation(inst_df: pd.DataFrame, out_path: Path) -> None:
    models        = sorted(inst_df["model_name"].unique())
    factors       = ["role", "fmt", "prefix"]
    factor_labels = {"role": "Role Framing", "fmt": "Format Directive", "prefix": "Answer Prefix"}
    eval_labels   = ["Heuristic", "Judge", "Heuristic−Judge\n(artifact share)"]

    fig, axes = _grid_axes(len(models), cell_w=5.0, cell_h=3.5, sharey=False)

    vmax = 0.20
    im   = None
    for ax, model in zip(axes, models):
        mdf  = inst_df[inst_df["model_name"] == model]
        data = np.zeros((len(factors), 3))

        for fi, factor in enumerate(factors):
            eh = float(mdf[f"effect_{factor}_heuristic"].mean())
            ej = float(mdf[f"effect_{factor}_judge"].mean())
            data[fi, 0] = eh
            data[fi, 1] = ej
            data[fi, 2] = eh - ej

        im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(3))
        ax.set_xticklabels(eval_labels, fontsize=8)
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels([factor_labels[f] for f in factors], fontsize=9)

        short_model = model.split("/")[-1]
        ax.set_title(short_model, fontsize=10, fontweight="bold", pad=6)

        for fi in range(len(factors)):
            for ei in range(3):
                val   = data[fi, ei]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(ei, fi, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    if im is not None:
        cbar = plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
        cbar.set_label("Effect on mean score\n(factor present − absent)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Figure 3  Structural Factor Effects\n"
        "(positive Heuristic−Judge = factor inflates apparent sensitivity)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Figure 4 : EAS by task type

def figure4_eas_by_task(inst_df: pd.DataFrame, out_path: Path) -> None:
    models  = sorted(inst_df["model_name"].unique())
    tasks   = ["mcq", "boolean", "open_ended"]
    t_label = {"mcq": "MCQ", "boolean": "Boolean", "open_ended": "Open-ended"}

    fig, axes = _grid_axes(len(models), cell_w=5.0, cell_h=4.0)

    for ax, model in zip(axes, models):
        mdf  = inst_df[inst_df["model_name"] == model]
        data = [mdf.loc[mdf["task_type"] == t, "eas"].dropna().values for t in tasks]
        bp   = ax.boxplot(data, patch_artist=True, widths=0.5,
                          medianprops={"color": "black", "linewidth": 2})
        colors = ["#5480E0", "#54C278", "#E07B54"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(tasks) + 1))
        ax.set_xticklabels([t_label[t] for t in tasks], fontsize=9)
        short_model = model.split("/")[-1]
        ax.set_title(short_model, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel("EAS (Evaluation-Attributable Sensitivity)", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Figure 4  EAS Distribution by Task Type",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Figure 5: Mean EAS with bootstrap CI by dataset × model


def figure5_eas_ci(agg_df: pd.DataFrame, out_path: Path) -> None:
    models   = sorted(agg_df["model_name"].unique())
    datasets = list(agg_df["dataset"].unique())
    has_ci   = "eas_ci_lo" in agg_df.columns

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(models)), 4.5))
    x       = np.arange(len(datasets))
    width   = 0.7 / len(models)
    colors  = ["#5480E0", "#E07B54", "#54C278", "#9B59B6",
               "#E0A020", "#20C0C0", "#C05080", "#80A040"]

    for mi, model in enumerate(models):
        mdf                  = agg_df[agg_df["model_name"] == model]
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

        offset     = (mi - (len(models) - 1) / 2) * (width + 0.02)
        short_name = model.split("/")[-1]
        bar_kw: dict = dict(label=short_name, color=colors[mi % len(colors)],
                            alpha=0.85, edgecolor="white", linewidth=0.8)
        if has_ci:
            bar_kw["yerr"]      = [lo_errs, hi_errs]
            bar_kw["capsize"]   = 3
            bar_kw["error_kw"]  = {"linewidth": 1.2, "ecolor": "black"}
        ax.bar(x + offset, means, width, **bar_kw)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=9)
    ax.set_ylabel("Mean EAS", fontsize=9)
    ci_note = "  (error bars = 95% bootstrap CI)" if has_ci else ""
    ax.set_title(f"Figure 5  Mean EAS by Dataset and Model{ci_note}",
                 fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Figure 6: Signed EAS — direction of evaluation disagreement


def figure6_signed_eas(inst_df: pd.DataFrame, out_path: Path) -> None:
    if "signed_eas" not in inst_df.columns:
        print(f"  Skipping {out_path.name} — signed_eas column missing (re-run metrics.py)")
        return

    models   = sorted(inst_df["model_name"].unique())
    datasets = list(inst_df["dataset"].unique())

    fig, axes = _grid_axes(len(models), cell_w=5.0, cell_h=4.0)

    for ax, model in zip(axes, models):
        mdf    = inst_df[inst_df["model_name"] == model]
        x      = np.arange(len(datasets))
        means  = [float(mdf.loc[mdf["dataset"] == ds, "signed_eas"].mean())
                  for ds in datasets]
        colors = ["#E07B54" if m >= 0 else "#5480E0" for m in means]
        ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=9)
        short_model = model.split("/")[-1]
        ax.set_title(short_model, fontsize=10, fontweight="bold", pad=6)
        ax.set_ylabel("Mean Signed EAS", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [
        mpatches.Patch(color="#E07B54", label="Heuristic > Judge  (heuristic inflates)"),
        mpatches.Patch(color="#5480E0", label="Judge > Heuristic  (judge inflates)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.01), fontsize=9, frameon=False)
    fig.suptitle(
        "Figure 6  Signed EAS: Direction of Evaluation Disagreement\n"
        "(positive = heuristic overestimates sensitivity, negative = judge overestimates)",
        y=1.05, fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures.")
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
        print(f"  Warning: dataset CSV not found at {args.in_dataset} — skipping Fig 5")

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

