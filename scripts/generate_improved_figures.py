#!/usr/bin/env python3
"""
Generate improved, publication-quality figures for the TemporalFusion paper.
Targets: ablation_charades, ablation_tradeoff, efficiency_scatter,
         loss_contribution_charades, tc_heatmap
Design: generous spacing, no overlapping, crisp readability.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "axes.grid": False,
    "text.color": "#1F2937",
    "axes.labelcolor": "#374151",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
})

# ── Color Palette ─────────────────────────────────────────────────────────────
C = {
    "blue":      "#3B82F6",
    "blue_d":    "#1D4ED8",
    "blue_l":    "#DBEAFE",
    "amber":     "#F59E0B",
    "amber_d":   "#B45309",
    "amber_l":   "#FEF3C7",
    "emerald":   "#10B981",
    "emerald_d": "#047857",
    "emerald_l": "#D1FAE5",
    "rose":      "#F43F5E",
    "rose_d":    "#BE123C",
    "rose_l":    "#FFE4E6",
    "slate":     "#64748B",
    "slate_d":   "#334155",
    "slate_l":   "#F1F5F9",
    "violet":    "#8B5CF6",
    "violet_d":  "#6D28D9",
    "violet_l":  "#EDE9FE",
    "cyan":      "#06B6D4",
    "cyan_l":    "#CFFAFE",
    "white":     "#FFFFFF",
    "bg":        "#FAFBFC",
}

MODELS_4 = {
    "TemporalFusion (Ours)": {"color": C["blue"],    "edge": C["blue_d"],   "short": "TF (Ours)"},
    "DirectTransformer":     {"color": C["amber"],   "edge": C["amber_d"],  "short": "DirectTrans."},
    "TemporalSegment":       {"color": C["emerald"], "edge": C["emerald_d"],"short": "Temp. Seg."},
    "MeanPool":              {"color": C["rose"],    "edge": C["rose_d"],   "short": "MeanPool"},
}

ABLATION_STYLES = {
    "full":           {"color": C["blue"],    "edge": C["blue_d"],    "label": "Full Model"},
    "no_temporal":    {"color": C["amber"],   "edge": C["amber_d"],   "label": "w/o Temporal Contr."},
    "no_collapse":    {"color": C["emerald"], "edge": C["emerald_d"], "label": "w/o Collapse Prev."},
    "no_crossscale":  {"color": C["violet"],  "edge": C["violet_d"],  "label": "w/o Cross-Scale"},
    "cls_only":       {"color": C["slate"],   "edge": C["slate_d"],   "label": "Cls. Only"},
}

OUTDIR = Path(__file__).resolve().parent.parent / "report" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"


def load_json(path):
    with open(path) as f:
        return json.load(f)

def soft_grid(ax, axis="y", alpha=0.12):
    ax.grid(True, axis=axis, alpha=alpha, linewidth=0.5, linestyle="--", color="#94A3B8")


all_results       = load_json(RESULTS_DIR / "all_results.json")
ablation_thumos   = load_json(RESULTS_DIR / "ablation_thumos14.json")
ablation_charades = load_json(RESULTS_DIR / "ablation_charades.json")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Charades Ablation — Horizontal grouped bar chart
#    Much cleaner than vertical bars with rotated labels
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_charades():
    variants = list(ABLATION_STYLES.keys())
    vlabels  = [ABLATION_STYLES[v]["label"] for v in variants]
    vcolors  = [ABLATION_STYLES[v]["color"] for v in variants]
    vedges   = [ABLATION_STYLES[v]["edge"]  for v in variants]

    mAPs = [ablation_charades[v]["mAP"] * 100 for v in variants]
    tcs  = [ablation_charades[v]["temporal_consistency"] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True,
                                    gridspec_kw={"width_ratios": [1.1, 1]})
    fig.patch.set_facecolor(C["white"])

    y = np.arange(len(variants))
    bar_h = 0.52

    # ── Left panel: mAP ──
    soft_grid(ax1, axis="x")
    for i in range(len(variants)):
        ax1.barh(y[i], mAPs[i], height=bar_h, color=vcolors[i],
                 edgecolor=C["white"], linewidth=0.6, zorder=3, alpha=0.88)
        # Value label to the right of bar
        ax1.text(mAPs[i] + 0.25, y[i], f"{mAPs[i]:.1f}%",
                 va="center", ha="left", fontsize=8.5, fontweight="bold",
                 color=vedges[i])

    # Delta annotation: cross-scale impact
    cs_idx = variants.index("no_crossscale")
    full_idx = variants.index("full")
    delta = mAPs[full_idx] - mAPs[cs_idx]
    ax1.annotate("", xy=(mAPs[cs_idx], y[cs_idx] + 0.02),
                 xytext=(mAPs[full_idx], y[full_idx] + 0.02),
                 arrowprops=dict(arrowstyle="<->", color=C["violet_d"],
                                 lw=1.3, shrinkA=3, shrinkB=3,
                                 connectionstyle="arc3,rad=-0.35"))
    mid_y = (y[cs_idx] + y[full_idx]) / 2
    ax1.text(11.5, mid_y, f"$\\Delta$={delta:+.1f}pp",
             fontsize=7.5, fontweight="bold", color=C["violet_d"], va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.15", fc=C["violet_l"], ec="none", alpha=0.85))

    # Highlight Full Model row
    ax1.axhspan(y[full_idx] - 0.35, y[full_idx] + 0.35,
                color=C["blue_l"], alpha=0.3, zorder=0)

    ax1.set_xlim(10, 20)
    ax1.set_yticks(y)
    ax1.set_yticklabels(vlabels, fontsize=9)
    ax1.set_xlabel("mAP (%)", fontsize=10)
    ax1.set_title("Mean Average Precision", fontsize=11, fontweight="bold", pad=8)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(left=False)

    # ── Right panel: TC ──
    soft_grid(ax2, axis="x")
    for i in range(len(variants)):
        ax2.barh(y[i], tcs[i], height=bar_h, color=vcolors[i],
                 edgecolor=C["white"], linewidth=0.6, zorder=3, alpha=0.88)
        ax2.text(tcs[i] + 0.003, y[i], f"{tcs[i]:.3f}",
                 va="center", ha="left", fontsize=8.5, fontweight="bold",
                 color=vedges[i])

    ax2.axhspan(y[full_idx] - 0.35, y[full_idx] + 0.35,
                color=C["blue_l"], alpha=0.3, zorder=0)

    ax2.set_xlim(0.92, 1.02)
    ax2.set_xlabel("Temporal Consistency", fontsize=10)
    ax2.set_title("Temporal Consistency", fontsize=11, fontweight="bold", pad=8)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)

    fig.suptitle("Charades — Ablation Study (Multi-Label)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=1.2)
    fig.savefig(OUTDIR / "ablation_charades.pdf")
    fig.savefig(OUTDIR / "ablation_charades.png")
    plt.close(fig)
    print("  ✓ ablation_charades")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Accuracy–TC Trade-off (THUMOS-14 ablation scatter)
#    Clean scatter with non-overlapping labels using offset arrows
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_tradeoff():
    variants = list(ABLATION_STYLES.keys())
    top1 = [ablation_thumos[v]["top1_accuracy"] * 100 for v in variants]
    tc   = [ablation_thumos[v]["temporal_consistency"] * 100 for v in variants]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor(C["white"])
    soft_grid(ax, axis="both")

    # Plot points
    for i, v in enumerate(variants):
        s = ABLATION_STYLES[v]
        ax.scatter(top1[i], tc[i], s=180, c=s["color"], edgecolors=s["edge"],
                   linewidth=1.8, zorder=5, alpha=0.92)

    # Pre-computed non-overlapping label positions (manual offsets tuned for the data)
    # Data points:
    #   full:          (87.74, 98.92)
    #   no_temporal:   (92.92, 86.19)
    #   no_collapse:   (91.04, 98.93)  ← close to full in TC, close to no_crossscale in acc
    #   no_crossscale: (90.57, 99.86)
    #   cls_only:      (88.68, 95.47)
    label_config = {
        "full":          {"offset": (-14, -18), "ha": "right"},
        "no_temporal":   {"offset": (14,   0),  "ha": "left"},
        "no_collapse":   {"offset": (14, -18),  "ha": "left"},
        "no_crossscale": {"offset": (14,  10),  "ha": "left"},
        "cls_only":      {"offset": (-14, 10),  "ha": "right"},
    }

    for i, v in enumerate(variants):
        s = ABLATION_STYLES[v]
        cfg = label_config[v]
        ax.annotate(
            s["label"], (top1[i], tc[i]),
            xytext=cfg["offset"], textcoords="offset points",
            fontsize=8.5, fontweight="bold", ha=cfg["ha"], va="center",
            color=s["edge"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor=C["white"],
                      edgecolor=s["color"], alpha=0.9, linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color=s["color"], lw=0.8,
                            alpha=0.6, shrinkB=5),
        )

    # Shaded region for ideal corner
    ax.fill_between([92, 96], 98, 104, color=C["emerald_l"], alpha=0.2, zorder=0)
    ax.text(94, 101.5, "Ideal\nregion", fontsize=7.5, color=C["emerald_d"],
            ha="center", va="center", fontstyle="italic", alpha=0.6)

    ax.set_xlabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_ylabel("Temporal Consistency (×100)", fontsize=11)
    ax.set_xlim(85, 96)
    ax.set_ylim(83, 103)

    ax.set_title("THUMOS-14 — Accuracy vs. Temporal Consistency",
                 fontweight="bold", fontsize=12, pad=10)

    plt.tight_layout()
    fig.savefig(OUTDIR / "ablation_tradeoff.pdf")
    fig.savefig(OUTDIR / "ablation_tradeoff.png")
    plt.close(fig)
    print("  ✓ ablation_tradeoff")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Efficiency Scatter (params vs accuracy, bubble = TC)
#    Log-scale x-axis, generous spacing, no label collisions
# ══════════════════════════════════════════════════════════════════════════════
def fig_efficiency_scatter():
    data = all_results["thumos14"]
    model_keys = ["TemporalFusion (Ours)", "DirectTransformer", "TemporalSegment", "MeanPool"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor(C["white"])
    soft_grid(ax, axis="both")

    # Pre-compute label positions to avoid overlap
    # Data: MeanPool (0.05, 87.74), TempSeg (2.1, 36.32),
    #       DirectTrans (77.7, 90.09), TF (77.9, 89.62)
    label_offsets = {
        "MeanPool":              {"ox": 14,  "oy":  0,  "ha": "left"},
        "TemporalSegment":       {"ox": 14,  "oy":  0,  "ha": "left"},
        "DirectTransformer":     {"ox": -14, "oy": 12,  "ha": "right"},
        "TemporalFusion (Ours)": {"ox": -14, "oy": -12, "ha": "right"},
    }

    for m in model_keys:
        d = data[m]
        s = MODELS_4[m]
        params_m = float(d["params"].replace("M", ""))
        top1 = d["top1"]
        tc = d["tc"]
        size = tc * 350 + 50

        # Outer glow
        ax.scatter(params_m, top1, s=size * 1.5, c=s["color"], alpha=0.07, zorder=2)
        # Main bubble
        ax.scatter(params_m, top1, s=size, c=s["color"], edgecolors=s["edge"],
                   linewidth=1.8, zorder=5, alpha=0.90)

        # Label
        lo = label_offsets[m]
        ax.annotate(
            f"{s['short']}\n({top1:.1f}%, TC={tc:.3f})",
            (params_m, top1),
            xytext=(lo["ox"], lo["oy"]), textcoords="offset points",
            fontsize=8, fontweight="bold", ha=lo["ha"], va="center",
            color=s["edge"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["white"],
                      edgecolor=s["color"], alpha=0.9, linewidth=0.7),
            arrowprops=dict(arrowstyle="-", color=s["color"], lw=0.7,
                            alpha=0.5, shrinkB=4),
        )

    ax.set_xlabel("Parameters (M)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 300)
    ax.set_ylim(28, 98)

    # Size legend
    for tc_val, label in [(0.80, "TC = 0.80"), (0.99, "TC = 0.99")]:
        ax.scatter([], [], s=tc_val * 350 + 50, c="#D1D5DB", edgecolors="#9CA3AF",
                   linewidth=1, label=label, alpha=0.5)
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.95, edgecolor="#E5E7EB",
                    title="Bubble size ∝ TC", title_fontsize=8,
                    handletextpad=0.8, borderpad=0.6)
    leg.get_title().set_fontweight("bold")

    ax.set_title("THUMOS-14 — Efficiency vs. Performance",
                 fontweight="bold", fontsize=12, pad=10)

    plt.tight_layout()
    fig.savefig(OUTDIR / "efficiency_scatter.pdf")
    fig.savefig(OUTDIR / "efficiency_scatter.png")
    plt.close(fig)
    print("  ✓ efficiency_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Loss Contribution Charades — Horizontal dumbbell / lollipop
#    Wider figure, value labels placed consistently, no overlap
# ══════════════════════════════════════════════════════════════════════════════
def fig_loss_contribution():
    full_map = ablation_charades["full"]["mAP"] * 100

    # Order by impact (largest delta first)
    components = [
        ("Cross-Scale Consistency",
         ablation_charades["no_crossscale"]["mAP"] * 100,
         C["violet"], C["violet_d"], C["violet_l"]),
        ("Collapse Prevention",
         ablation_charades["no_collapse"]["mAP"] * 100,
         C["emerald"], C["emerald_d"], C["emerald_l"]),
        ("Temporal Contrastive",
         ablation_charades["no_temporal"]["mAP"] * 100,
         C["amber"], C["amber_d"], C["amber_l"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 3.0))
    fig.patch.set_facecolor(C["white"])

    y_positions = np.arange(len(components))[::-1]

    for yi, (name, wo_val, col, col_d, col_l) in zip(y_positions, components):
        delta = full_map - wo_val
        lo_val, hi_val = min(wo_val, full_map), max(wo_val, full_map)

        # Connecting bar (thicker, more visible)
        ax.plot([lo_val, hi_val], [yi, yi], color=col, linewidth=5, alpha=0.18, zorder=2,
                solid_capstyle="round")

        # "Without" dot (open)
        ax.scatter(wo_val, yi, s=120, color=C["white"], edgecolors=col_d,
                   linewidth=2, zorder=6)

        # "Full model" dot (filled)
        ax.scatter(full_map, yi, s=120, color=col, edgecolors=col_d,
                   linewidth=1.5, zorder=6)

        # Value labels: "without" value goes to the left of the leftmost dot,
        # "full" value goes to the right of the rightmost dot.
        # Handle both cases: wo_val < full_map AND wo_val > full_map
        if wo_val <= full_map:
            ax.text(wo_val - 0.2, yi, f"{wo_val:.1f}%", ha="right", va="center",
                    fontsize=8.5, color=col_d, fontweight="bold")
            ax.text(full_map + 0.2, yi, f"{full_map:.1f}%", ha="left", va="center",
                    fontsize=8.5, color=C["blue_d"], fontweight="bold")
        else:
            ax.text(full_map - 0.2, yi, f"{full_map:.1f}%", ha="right", va="center",
                    fontsize=8.5, color=C["blue_d"], fontweight="bold")
            ax.text(wo_val + 0.2, yi, f"{wo_val:.1f}%", ha="left", va="center",
                    fontsize=8.5, color=col_d, fontweight="bold")

        # Delta label (centered above the connecting line)
        mid = (wo_val + full_map) / 2
        sign = "+" if delta >= 0 else ""
        ax.text(mid, yi + 0.32, f"$\\Delta$ = {sign}{delta:.1f} pp",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=col_d,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=col_l, edgecolor="none", alpha=0.85))

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in components], fontsize=9.5)
    ax.set_xlabel("mAP (%)", fontsize=10)
    ax.set_xlim(11, 19)
    ax.set_ylim(-0.6, len(components) - 0.3)
    soft_grid(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Legend
    ax.scatter([], [], s=70, color=C["white"], edgecolors=C["slate_d"],
               linewidth=1.5, label="Without component")
    ax.scatter([], [], s=70, color=C["slate"], edgecolors=C["slate_d"],
               linewidth=1.2, label="Full model")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95, edgecolor="#E5E7EB",
              handletextpad=0.4, borderpad=0.5)

    ax.set_title("Charades — Impact of Each Loss Component on mAP",
                 fontweight="bold", fontsize=12, pad=10)

    plt.tight_layout()
    fig.savefig(OUTDIR / "loss_contribution_charades.pdf")
    fig.savefig(OUTDIR / "loss_contribution_charades.png")
    plt.close(fig)
    print("  ✓ loss_contribution_charades")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TC Heatmap — Clean 2×4 grid with better spacing and readability
# ══════════════════════════════════════════════════════════════════════════════
def fig_tc_heatmap():
    model_keys = ["TemporalFusion (Ours)", "DirectTransformer", "TemporalSegment", "MeanPool"]
    model_labels = ["TF\n(Ours)", "Direct\nTrans.", "Temp.\nSeg.", "Mean\nPool"]
    datasets = ["THUMOS-14", "Charades"]

    tc_thumos   = [all_results["thumos14"][m]["tc"] for m in model_keys]
    tc_charades = [all_results["charades"][m]["tc"] for m in model_keys]
    data = np.array([tc_thumos, tc_charades])

    # Custom diverging-ish colormap: red-ish low → green-ish mid → blue high
    colors_list = ["#FCA5A5", "#FDE68A", "#6EE7B7", "#93C5FD", "#3B82F6"]
    cmap = LinearSegmentedColormap.from_list("tc_cmap", colors_list, N=256)

    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    fig.patch.set_facecolor(C["white"])

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.60, vmax=1.01)

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=9, linespacing=1.1)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10, fontweight="bold")
    ax.tick_params(length=0, pad=6)
    ax.xaxis.set_ticks_position("top")

    # Cell text with smart contrast
    for i in range(len(datasets)):
        for j in range(len(model_labels)):
            val = data[i, j]
            # White text with dark stroke on high-TC (dark blue) cells,
            # dark text on low-TC (light) cells
            if val > 0.95:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=C["white"],
                        path_effects=[pe.withStroke(linewidth=2.5, foreground=C["blue_d"])])
            else:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=15, fontweight="bold", color=C["slate_d"])

    # Clean borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White grid lines to separate cells
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=C["white"], linewidth=4)
    ax.tick_params(which="minor", length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.04, aspect=10)
    cbar.set_label("TC", fontsize=9, labelpad=4)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title("Temporal Consistency Across Models & Datasets",
                 fontweight="bold", fontsize=11, pad=14)

    plt.tight_layout()
    fig.savefig(OUTDIR / "tc_heatmap.pdf")
    fig.savefig(OUTDIR / "tc_heatmap.png")
    plt.close(fig)
    print("  ✓ tc_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating improved figures ...")
    fig_ablation_charades()
    fig_ablation_tradeoff()
    fig_efficiency_scatter()
    fig_loss_contribution()
    fig_tc_heatmap()
    print(f"\nAll 5 figures saved to {OUTDIR}/")
