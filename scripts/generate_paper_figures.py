#!/usr/bin/env python3
"""
Generate modern, publication-quality figures for the TemporalFusion paper.
Design: clean minimalism, soft gradients, rounded elements, generous spacing.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "lines.linewidth": 2,
    "patch.linewidth": 0,
    "text.color": "#1F2937",
    "axes.labelcolor": "#374151",
    "xtick.color": "#6B7280",
    "ytick.color": "#6B7280",
})

# ── Color Palette — refined, accessible ───────────────────────────────────────
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

# Model styling
MODELS_4 = {
    "TemporalFusion (Ours)": {"color": C["blue"],    "edge": C["blue_d"],   "label": "TemporalFusion (Ours)", "short": "TF (Ours)"},
    "DirectTransformer":     {"color": C["amber"],   "edge": C["amber_d"],  "label": "DirectTransformer",     "short": "DirectTrans."},
    "TemporalSegment":       {"color": C["emerald"], "edge": C["emerald_d"],"label": "TemporalSegment",       "short": "Temp. Seg."},
    "MeanPool":              {"color": C["rose"],    "edge": C["rose_d"],   "label": "MeanPool",              "short": "MeanPool"},
}

ABLATION_STYLES = {
    "full":           {"color": C["blue"],    "edge": C["blue_d"],    "label": "Full Model"},
    "no_temporal":    {"color": C["amber"],   "edge": C["amber_d"],   "label": "w/o Temporal Contr."},
    "no_collapse":    {"color": C["emerald"], "edge": C["emerald_d"], "label": "w/o Collapse Prev."},
    "no_crossscale":  {"color": C["violet"],  "edge": C["violet_d"],  "label": "w/o Cross-Scale"},
    "cls_only":       {"color": C["slate"],   "edge": C["slate_d"],   "label": "Classification Only"},
}

OUTDIR = Path(__file__).resolve().parent.parent / "report" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def add_value_label(ax, x, y, text, color="#374151", fontsize=8.5, offset=(0, 6),
                    ha="center", bold=True, bg=None):
    weight = "bold" if bold else "normal"
    txt = ax.annotate(text, (x, y), xytext=offset, textcoords="offset points",
                      ha=ha, va="bottom", fontsize=fontsize, fontweight=weight,
                      color=color, zorder=10)
    if bg:
        txt.set_bbox(dict(boxstyle="round,pad=0.15", facecolor=bg, edgecolor="none", alpha=0.7))
    return txt

def soft_grid(ax, axis="y", alpha=0.12):
    ax.grid(True, axis=axis, alpha=alpha, linewidth=0.5, linestyle="--", color="#94A3B8")


RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"
all_results       = load_json(RESULTS_DIR / "all_results.json")
ablation_thumos   = load_json(RESULTS_DIR / "ablation_thumos14.json")
ablation_charades = load_json(RESULTS_DIR / "ablation_charades.json")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — THUMOS-14 Main Comparison (horizontal bars, 3-panel)
# ══════════════════════════════════════════════════════════════════════════════
def fig_thumos14_comparison():
    data = all_results["thumos14"]
    models = ["MeanPool", "TemporalSegment", "DirectTransformer", "TemporalFusion (Ours)"]
    top1 = [data[m]["top1"] for m in models]
    top5 = [data[m]["top5"] for m in models]
    tc   = [data[m]["tc"]   for m in models]
    colors = [MODELS_4[m]["color"] for m in models]
    edges  = [MODELS_4[m]["edge"]  for m in models]
    labels = [MODELS_4[m]["short"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    fig.patch.set_facecolor(C["white"])

    metrics = [("Top-1 Accuracy (%)", top1, 30, 95),
               ("Top-5 Accuracy (%)", top5, 60, 102),
               ("Temporal Consistency", tc,  0.6, 1.05)]

    y = np.arange(len(models))

    for ax, (title, vals, lo, hi) in zip(axes, metrics):
        soft_grid(ax, axis="x")
        for i, (v, c, e) in enumerate(zip(vals, colors, edges)):
            ax.barh(i, v, height=0.52, color=c, edgecolor=C["white"],
                    linewidth=0.8, zorder=3, alpha=0.88)
            ax.plot(v, i, "o", color=e, markersize=6, zorder=5,
                    markeredgecolor=C["white"], markeredgewidth=1.2)
            fmt = f"{v:.3f}" if title.startswith("Temporal") else f"{v:.1f}"
            ax.text(v + (hi - lo) * 0.03, i, fmt,
                    va="center", ha="left", fontsize=8.5, fontweight="bold", color=e)

        ax.set_xlim(lo, hi)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    for ax in axes:
        ax.axhspan(len(models) - 1 - 0.38, len(models) - 1 + 0.38,
                   color=C["blue_l"], alpha=0.35, zorder=0)

    fig.suptitle("THUMOS-14 — Model Comparison", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout(w_pad=1.8)
    fig.savefig(OUTDIR / "thumos14_comparison.pdf")
    fig.savefig(OUTDIR / "thumos14_comparison.png")
    plt.close(fig)
    print("  ✓ thumos14_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Charades Main Comparison (horizontal bars, 2-panel)
# ══════════════════════════════════════════════════════════════════════════════
def fig_charades_comparison():
    data = all_results["charades"]
    models = ["MeanPool", "TemporalSegment", "DirectTransformer", "TemporalFusion (Ours)"]
    mAP = [data[m]["mAP"] for m in models]
    tc  = [data[m]["tc"]  for m in models]
    colors = [MODELS_4[m]["color"] for m in models]
    edges  = [MODELS_4[m]["edge"]  for m in models]
    labels = [MODELS_4[m]["short"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    fig.patch.set_facecolor(C["white"])

    y = np.arange(len(models))

    for ax, vals, title, lo, hi, fmt_str in [
        (ax1, mAP, "mAP (%)", 12, 20, "{:.2f}"),
        (ax2, tc,  "Temporal Consistency", 0.55, 1.05, "{:.3f}"),
    ]:
        soft_grid(ax, axis="x")
        for i, (v, c, e) in enumerate(zip(vals, colors, edges)):
            ax.barh(i, v, height=0.50, color=c, edgecolor=C["white"],
                    linewidth=0.8, zorder=3, alpha=0.85)
            ax.plot(v, i, "o", color=e, markersize=6, zorder=5,
                    markeredgecolor=C["white"], markeredgewidth=1.2)
            ax.text(v + (hi - lo) * 0.03, i, fmt_str.format(v),
                    va="center", ha="left", fontsize=8.5, fontweight="bold", color=e)

        ax.set_xlim(lo, hi)
        ax.set_yticks(y)
        if ax is ax1:
            ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    for ax in [ax1, ax2]:
        ax.axhspan(len(models) - 1 - 0.38, len(models) - 1 + 0.38,
                   color=C["blue_l"], alpha=0.35, zorder=0)

    fig.suptitle("Charades — Model Comparison (Multi-Label)", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout(w_pad=2.0)
    fig.savefig(OUTDIR / "charades_comparison.pdf")
    fig.savefig(OUTDIR / "charades_comparison.png")
    plt.close(fig)
    print("  ✓ charades_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — THUMOS-14 Ablation (paired vertical bars)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_thumos():
    variants = list(ABLATION_STYLES.keys())
    vlabels  = [ABLATION_STYLES[v]["label"] for v in variants]
    vcolors  = [ABLATION_STYLES[v]["color"] for v in variants]
    vedges   = [ABLATION_STYLES[v]["edge"]  for v in variants]

    top1 = [ablation_thumos[v]["top1_accuracy"] * 100 for v in variants]
    tc   = [ablation_thumos[v]["temporal_consistency"] * 100 for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    fig.patch.set_facecolor(C["white"])

    x = np.arange(len(variants))
    w = 0.56

    # Top-1 Accuracy
    soft_grid(ax1)
    bars1 = ax1.bar(x, top1, w, color=vcolors, edgecolor=[C["white"]] * len(variants),
                    linewidth=1.0, zorder=3, alpha=0.88)
    ax1.set_ylim(82, 100)
    for i, (b, v) in enumerate(zip(bars1, top1)):
        add_value_label(ax1, b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        color=vedges[i], fontsize=8.5, offset=(0, 4))
    ax1.set_xticks(x)
    ax1.set_xticklabels(vlabels, fontsize=8, rotation=25, ha="right")
    ax1.set_ylabel("Top-1 Accuracy (%)")
    ax1.set_title("Top-1 Accuracy", fontsize=11, fontweight="bold", pad=10)
    ax1.axhspan(top1[0] - 0.5, top1[0] + 0.5, color=C["blue_l"], alpha=0.25, zorder=0)

    # Temporal Consistency
    soft_grid(ax2)
    bars2 = ax2.bar(x, tc, w, color=vcolors, edgecolor=[C["white"]] * len(variants),
                    linewidth=1.0, zorder=3, alpha=0.88)
    ax2.set_ylim(82, 104)
    for i, (b, v) in enumerate(zip(bars2, tc)):
        add_value_label(ax2, b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        color=vedges[i], fontsize=8.5, offset=(0, 4))
    ax2.set_xticks(x)
    ax2.set_xticklabels(vlabels, fontsize=8, rotation=25, ha="right")
    ax2.set_ylabel("Temporal Consistency (x100)")
    ax2.set_title("Temporal Consistency", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle("THUMOS-14 — Ablation Study", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=2.5)
    fig.savefig(OUTDIR / "ablation_thumos14.pdf")
    fig.savefig(OUTDIR / "ablation_thumos14.png")
    plt.close(fig)
    print("  ✓ ablation_thumos14")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Charades Ablation (paired bars: mAP + TC)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_charades():
    variants = list(ABLATION_STYLES.keys())
    vlabels  = [ABLATION_STYLES[v]["label"] for v in variants]
    vcolors  = [ABLATION_STYLES[v]["color"] for v in variants]
    vedges   = [ABLATION_STYLES[v]["edge"]  for v in variants]

    mAPs = [ablation_charades[v]["mAP"] * 100 for v in variants]
    tcs  = [ablation_charades[v]["temporal_consistency"] * 100 for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    fig.patch.set_facecolor(C["white"])
    x = np.arange(len(variants))
    w = 0.56

    # mAP
    soft_grid(ax1)
    bars1 = ax1.bar(x, mAPs, w, color=vcolors, edgecolor=[C["white"]] * len(variants),
                    linewidth=1.0, zorder=3, alpha=0.88)
    ax1.set_ylim(10, 21)
    for i, (b, v) in enumerate(zip(bars1, mAPs)):
        add_value_label(ax1, b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        color=vedges[i], fontsize=8.5, offset=(0, 4))
    ax1.set_xticks(x)
    ax1.set_xticklabels(vlabels, fontsize=8, rotation=25, ha="right")
    ax1.set_ylabel("mAP (%)")
    ax1.set_title("Mean Average Precision", fontsize=11, fontweight="bold", pad=10)

    # Delta annotation for cross-scale
    cs_idx = variants.index("no_crossscale")
    full_idx = variants.index("full")
    delta = mAPs[full_idx] - mAPs[cs_idx]
    mid_y = (mAPs[full_idx] + mAPs[cs_idx]) / 2
    ax1.annotate("", xy=(cs_idx, mAPs[cs_idx] + 0.3), xytext=(cs_idx, mAPs[full_idx] - 0.3),
                 arrowprops=dict(arrowstyle="<->", color=C["violet_d"], lw=1.5, shrinkA=0, shrinkB=0))
    ax1.text(cs_idx + 0.42, mid_y, f"$\\Delta$={delta:+.1f}",
             fontsize=8, fontweight="bold", color=C["violet_d"], va="center",
             bbox=dict(boxstyle="round,pad=0.15", fc=C["violet_l"], ec="none", alpha=0.8))

    # Temporal Consistency
    soft_grid(ax2)
    bars2 = ax2.bar(x, tcs, w, color=vcolors, edgecolor=[C["white"]] * len(variants),
                    linewidth=1.0, zorder=3, alpha=0.88)
    ax2.set_ylim(92, 102)
    for i, (b, v) in enumerate(zip(bars2, tcs)):
        add_value_label(ax2, b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                        color=vedges[i], fontsize=8.5, offset=(0, 4))
    ax2.set_xticks(x)
    ax2.set_xticklabels(vlabels, fontsize=8, rotation=25, ha="right")
    ax2.set_ylabel("Temporal Consistency (x100)")
    ax2.set_title("Temporal Consistency", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle("Charades — Ablation Study (Multi-Label)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout(w_pad=2.5)
    fig.savefig(OUTDIR / "ablation_charades.pdf")
    fig.savefig(OUTDIR / "ablation_charades.png")
    plt.close(fig)
    print("  ✓ ablation_charades")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — TC Heatmap (refined with custom colormap)
# ══════════════════════════════════════════════════════════════════════════════
def fig_tc_heatmap():
    model_keys = ["TemporalFusion (Ours)", "DirectTransformer", "TemporalSegment", "MeanPool"]
    model_labels = ["TF (Ours)", "DirectTrans.", "Temp. Seg.", "MeanPool"]
    datasets = ["THUMOS-14", "Charades"]

    tc_thumos   = [all_results["thumos14"][m]["tc"] for m in model_keys]
    tc_charades = [all_results["charades"][m]["tc"] for m in model_keys]
    data = np.array([tc_thumos, tc_charades])

    colors_list = ["#FEE2E2", "#FEF3C7", "#D1FAE5", "#DBEAFE", "#3B82F6"]
    cmap = LinearSegmentedColormap.from_list("tc_cmap", colors_list)

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    fig.patch.set_facecolor(C["white"])

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.55, vmax=1.02)

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10.5, fontweight="bold")
    ax.tick_params(length=0)

    for i in range(len(datasets)):
        for j in range(len(model_labels)):
            val = data[i, j]
            text_color = C["white"] if val > 0.92 else C["slate_d"]
            effects = [pe.withStroke(linewidth=2.5, foreground=C["white"])] if val > 0.92 else []
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=text_color,
                    path_effects=effects)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=C["white"], linewidth=3)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=12)
    cbar.set_label("Temporal Consistency", fontsize=10)
    cbar.outline.set_visible(False)

    ax.set_title("Temporal Consistency Across Models & Datasets",
                 fontweight="bold", fontsize=12, pad=12)

    plt.tight_layout()
    fig.savefig(OUTDIR / "tc_heatmap.pdf")
    fig.savefig(OUTDIR / "tc_heatmap.png")
    plt.close(fig)
    print("  ✓ tc_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Loss Contribution (Charades — dumbbell chart)
# ══════════════════════════════════════════════════════════════════════════════
def fig_loss_contribution():
    full_map = ablation_charades["full"]["mAP"] * 100
    components = [
        ("Cross-Scale\nConsistency", ablation_charades["no_crossscale"]["mAP"] * 100, C["violet"], C["violet_d"], C["violet_l"]),
        ("Collapse\nPrevention", ablation_charades["no_collapse"]["mAP"] * 100, C["emerald"], C["emerald_d"], C["emerald_l"]),
        ("Temporal\nContrastive", ablation_charades["no_temporal"]["mAP"] * 100, C["amber"], C["amber_d"], C["amber_l"]),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    fig.patch.set_facecolor(C["white"])

    y_positions = np.arange(len(components))[::-1]

    for yi, (name, wo_val, col, col_d, col_l) in zip(y_positions, components):
        delta = full_map - wo_val

        # Connecting line
        lo_val, hi_val = min(wo_val, full_map), max(wo_val, full_map)
        ax.plot([lo_val, hi_val], [yi, yi], color=col, linewidth=3, alpha=0.3, zorder=2)

        # "Without" dot
        ax.scatter(wo_val, yi, s=140, color=col_l, edgecolors=col_d, linewidth=1.5, zorder=5)
        ax.text(wo_val, yi + 0.30, f"{wo_val:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=col_d, fontweight="bold")

        # "Full model" dot
        ax.scatter(full_map, yi, s=140, color=col, edgecolors=col_d, linewidth=1.5, zorder=5)
        ax.text(full_map, yi + 0.30, f"{full_map:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=C["blue_d"], fontweight="bold")

        # Delta label
        mid = (wo_val + full_map) / 2
        sign = "+" if delta >= 0 else ""
        ax.text(mid, yi - 0.35, f"$\\Delta$ = {sign}{delta:.1f} pp",
                ha="center", va="top", fontsize=8.5, fontweight="bold", color=col_d,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=col_l, edgecolor="none", alpha=0.85))

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in components], fontsize=9.5)
    ax.set_xlabel("mAP (%)", fontsize=11)
    ax.set_xlim(11, 19)
    soft_grid(ax, axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    ax.scatter([], [], s=80, color=C["slate_l"], edgecolors=C["slate_d"], linewidth=1.2, label="Without component")
    ax.scatter([], [], s=80, color=C["blue"],    edgecolors=C["blue_d"],  linewidth=1.2, label="Full model")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#E5E7EB",
              handletextpad=0.4, borderpad=0.5)

    ax.set_title("Charades — Impact of Each Loss Component on mAP",
                 fontweight="bold", fontsize=12, pad=12)

    plt.tight_layout()
    fig.savefig(OUTDIR / "loss_contribution_charades.pdf")
    fig.savefig(OUTDIR / "loss_contribution_charades.png")
    plt.close(fig)
    print("  ✓ loss_contribution_charades")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Efficiency Scatter (params vs accuracy, bubble = TC)
# ══════════════════════════════════════════════════════════════════════════════
def fig_efficiency_scatter():
    data = all_results["thumos14"]
    model_keys = ["TemporalFusion (Ours)", "DirectTransformer", "TemporalSegment", "MeanPool"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(C["white"])
    soft_grid(ax, axis="both")

    for m in model_keys:
        d = data[m]
        s = MODELS_4[m]
        params_m = float(d["params"].replace("M", ""))
        top1 = d["top1"]
        tc = d["tc"]
        size = tc * 400 + 30

        # Glow effect
        ax.scatter(params_m, top1, s=size * 1.6, c=s["color"], alpha=0.08, zorder=2)
        ax.scatter(params_m, top1, s=size, c=s["color"], edgecolors=s["edge"],
                   linewidth=1.8, zorder=5, alpha=0.90)

        # Smart label placement
        if "MeanPool" in m:
            ox, oy, ha = -8, -14, "right"
        elif "Temporal" in m and "Fusion" not in m:
            ox, oy, ha = 8, -12, "left"
        elif "Direct" in m:
            ox, oy, ha = 8, 8, "left"
        else:
            ox, oy, ha = 8, 8, "left"

        label_text = s["short"]
        ax.annotate(label_text, (params_m, top1),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=9, fontweight="bold", ha=ha, va="center",
                    color=s["edge"],
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=C["white"],
                              edgecolor=s["color"], alpha=0.85, linewidth=0.8))

    ax.set_xlabel("Parameters (M)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 250)
    ax.set_ylim(25, 98)

    for tc_val, label in [(0.80, "TC = 0.80"), (0.99, "TC = 0.99")]:
        ax.scatter([], [], s=tc_val * 400 + 30, c="#D1D5DB", edgecolors="#9CA3AF",
                   linewidth=1, label=label, alpha=0.6)
    leg = ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="#E5E7EB",
                    title="Bubble size $\\propto$ TC", title_fontsize=8.5,
                    handletextpad=0.8, borderpad=0.6)
    leg.get_title().set_fontweight("bold")

    ax.set_title("THUMOS-14 — Efficiency vs. Performance",
                 fontweight="bold", fontsize=12, pad=12)

    plt.tight_layout()
    fig.savefig(OUTDIR / "efficiency_scatter.pdf")
    fig.savefig(OUTDIR / "efficiency_scatter.png")
    plt.close(fig)
    print("  ✓ efficiency_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Accuracy–TC Trade-off (THUMOS-14 ablation, scatter)
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation_tradeoff():
    variants = list(ABLATION_STYLES.keys())
    top1 = [ablation_thumos[v]["top1_accuracy"] * 100 for v in variants]
    tc   = [ablation_thumos[v]["temporal_consistency"] * 100 for v in variants]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor(C["white"])
    soft_grid(ax, axis="both")

    for i, v in enumerate(variants):
        s = ABLATION_STYLES[v]
        ax.scatter(top1[i], tc[i], s=280, c=s["color"], alpha=0.12, zorder=2)
        ax.scatter(top1[i], tc[i], s=160, c=s["color"], edgecolors=s["edge"],
                   linewidth=2, zorder=5, alpha=0.92)

    offsets = {
        "full":          (12, 6),
        "no_temporal":   (10, 8),
        "no_collapse":   (10, -14),
        "no_crossscale": (-12, 8),
        "cls_only":      (12, -10),
    }
    ha_map = {
        "full": "left", "no_temporal": "left", "no_collapse": "left",
        "no_crossscale": "right", "cls_only": "left",
    }

    for i, v in enumerate(variants):
        s = ABLATION_STYLES[v]
        ox, oy = offsets[v]
        ax.annotate(
            s["label"], (top1[i], tc[i]),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=8.5, fontweight="bold", ha=ha_map[v], va="center",
            color=s["edge"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C["white"],
                      edgecolor=s["color"], alpha=0.85, linewidth=0.8),
            arrowprops=dict(arrowstyle="-", color=s["color"], lw=0.8, alpha=0.5),
        )

    ax.annotate("Ideal region\n(high acc. + high TC)", xy=(93, 101),
                fontsize=7.5, color="#9CA3AF", ha="center", va="center",
                fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["slate_l"],
                          edgecolor="none", alpha=0.5))

    ax.set_xlabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_ylabel("Temporal Consistency (x100)", fontsize=11)
    ax.set_xlim(84, 96)
    ax.set_ylim(82, 104)

    ax.set_title("THUMOS-14 — Accuracy vs. Temporal Consistency Trade-off",
                 fontweight="bold", fontsize=12, pad=12)

    plt.tight_layout()
    fig.savefig(OUTDIR / "ablation_tradeoff.pdf")
    fig.savefig(OUTDIR / "ablation_tradeoff.png")
    plt.close(fig)
    print("  ✓ ablation_tradeoff")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Architecture Diagram (modern block flow)
# ══════════════════════════════════════════════════════════════════════════════
def fig_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor(C["white"])
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.8, 4.2)
    ax.axis("off")

    blocks = [
        {"x": 0.6,  "y": 2.3, "w": 1.5, "h": 1.3,
         "label": "Video\nFeatures",
         "sub": "(B, T, D)",
         "fc": C["slate_l"], "ec": "#94A3B8"},
        {"x": 2.8,  "y": 2.3, "w": 1.5, "h": 1.3,
         "label": "Feature\nProjection",
         "sub": "Linear + LN + GELU",
         "fc": C["blue_l"], "ec": C["blue"]},
        {"x": 5.0,  "y": 2.3, "w": 1.5, "h": 1.3,
         "label": "Transformer\nEncoder",
         "sub": "6L x 8H, Pre-LN",
         "fc": C["amber_l"], "ec": C["amber"]},
        {"x": 7.2,  "y": 2.3, "w": 1.8, "h": 1.3,
         "label": "Hierarchical\nAggregation",
         "sub": "4 levels, /16",
         "fc": C["emerald_l"], "ec": C["emerald"]},
        {"x": 9.7,  "y": 2.3, "w": 1.3, "h": 1.3,
         "label": "Classifier",
         "sub": "LN + Linear + C",
         "fc": C["rose_l"], "ec": C["rose"]},
    ]

    for blk in blocks:
        rect = FancyBboxPatch(
            (blk["x"] - blk["w"] / 2, blk["y"] - blk["h"] / 2),
            blk["w"], blk["h"],
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=blk["fc"], edgecolor=blk["ec"], linewidth=2,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(blk["x"], blk["y"] + 0.12, blk["label"],
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color=C["slate_d"], zorder=4)
        ax.text(blk["x"], blk["y"] - 0.4, blk["sub"],
                ha="center", va="center", fontsize=7, color="#6B7280",
                fontstyle="italic", zorder=4)

    arrow_kw = dict(arrowstyle="-|>", color="#475569", lw=2,
                    connectionstyle="arc3,rad=0", mutation_scale=14)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for i, j in pairs:
        x1 = blocks[i]["x"] + blocks[i]["w"] / 2 + 0.08
        x2 = blocks[j]["x"] - blocks[j]["w"] / 2 - 0.08
        ax.annotate("", xy=(x2, 2.3), xytext=(x1, 2.3),
                    arrowprops=arrow_kw, zorder=4)

    # Positional encoding badge
    pe_x = (blocks[1]["x"] + blocks[2]["x"]) / 2
    pe_rect = FancyBboxPatch(
        (pe_x - 0.55, 3.25), 1.1, 0.5,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=C["cyan_l"], edgecolor=C["cyan"], linewidth=1.2, zorder=5,
    )
    ax.add_patch(pe_rect)
    ax.text(pe_x, 3.50, "+ Pos. Enc.", ha="center", va="center",
            fontsize=8, fontweight="bold", color="#0E7490", zorder=6)
    ax.annotate("", xy=(pe_x, 2.95), xytext=(pe_x, 3.22),
                arrowprops=dict(arrowstyle="-|>", color=C["cyan"], lw=1.2, mutation_scale=10),
                zorder=5)

    # Loss labels below
    losses = [
        {"x": 5.0,  "label": "$\\mathcal{L}_{tc}$ + $\\mathcal{L}_{reg}$",
         "desc": "Temporal Contr. +\nCollapse Prev.",
         "color": C["amber"], "color_d": C["amber_d"]},
        {"x": 7.2,  "label": "$\\mathcal{L}_{cs}$",
         "desc": "Cross-Scale\nConsistency",
         "color": C["emerald"], "color_d": C["emerald_d"]},
        {"x": 9.7,  "label": "$\\mathcal{L}_{cls}$",
         "desc": "Classification",
         "color": C["rose"], "color_d": C["rose_d"]},
    ]

    for loss in losses:
        ax.annotate("", xy=(loss["x"], blocks[0]["y"] - blocks[0]["h"] / 2 - 0.05),
                    xytext=(loss["x"], 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=loss["color"],
                                    lw=1.5, linestyle="--", mutation_scale=10),
                    zorder=4)
        ax.text(loss["x"], 0.25, loss["label"],
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=loss["color_d"], zorder=6)
        ax.text(loss["x"], -0.2, loss["desc"],
                ha="center", va="center", fontsize=7, color="#6B7280",
                fontstyle="italic", zorder=6)

    # Pyramid visualization inside hierarchy block
    pyr_x = 7.2
    for lvl, (dy, pw) in enumerate([(0.4, 0.7), (0.15, 0.5), (-0.1, 0.3), (-0.3, 0.15)]):
        rect = FancyBboxPatch(
            (pyr_x - pw / 2, 2.3 + dy - 0.06), pw, 0.12,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=C["emerald"], edgecolor=C["white"], linewidth=0.5,
            alpha=0.35 + lvl * 0.12, zorder=4,
        )
        ax.add_patch(rect)

    ax.text(5.5, 4.05, "TemporalFusion  —  Architecture Overview",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["slate_d"])

    plt.tight_layout(pad=0.3)
    fig.savefig(OUTDIR / "architecture_diagram.pdf")
    fig.savefig(OUTDIR / "architecture_diagram.png")
    plt.close(fig)
    print("  ✓ architecture_diagram")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures ...")
    fig_thumos14_comparison()
    fig_charades_comparison()
    fig_ablation_thumos()
    fig_ablation_charades()
    fig_tc_heatmap()
    fig_loss_contribution()
    fig_efficiency_scatter()
    fig_ablation_tradeoff()
    fig_architecture_diagram()
    print(f"\nAll 9 figures saved to {OUTDIR}/")
