# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import os

# Output directory (same as script location)
out_dir = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Model definitions and colors
# -----------------------------
models = ["GPT-4o", "Llama 3.1 70B", "Llama 3.1 8B", "o1"]

# Saturated = SHORT, Pastel = LONG
color_map = {
    "GPT-4o":         {"SHORT": "#548235", "LONG": "#C5E0B4"},
    "Llama 3.1 70B":  {"SHORT": "#BF9000", "LONG": "#FFE699"},
    "Llama 3.1 8B":   {"SHORT": "#C55A11", "LONG": "#F8CBAD"},
    "o1":             {"SHORT": "#2E75B6", "LONG": "#BDD7EE"},
}

# -----------------------------
# Hardcoded data from evalLongShortOutput.txt
# -----------------------------
accuracy = {
    "GPT-4o":         {"LONG": 0.350, "SHORT": 0.740},
    "Llama 3.1 70B":  {"LONG": 0.475, "SHORT": 0.660},
    "Llama 3.1 8B":   {"LONG": 0.300, "SHORT": 0.320},
    "o1":             {"LONG": 0.575, "SHORT": 0.840},
}

auroc = {
    "GPT-4o":         {"LONG": 0.723, "SHORT": 0.599},
    "Llama 3.1 70B":  {"LONG": 0.799, "SHORT": 0.791},
    "Llama 3.1 8B":   {"LONG": 0.723, "SHORT": 0.735},
    "o1":             {"LONG": 0.614, "SHORT": 0.708},
}

aurac = {
    "GPT-4o":         {"LONG": 0.477, "SHORT": 0.854},
    "Llama 3.1 70B":  {"LONG": 0.675, "SHORT": 0.848},
    "Llama 3.1 8B":   {"LONG": 0.485, "SHORT": 0.474},
    "o1":             {"LONG": 0.731, "SHORT": 0.908},
}

# RAC data: rejection accuracy at 10%, 20%, ..., 100%
rac_pct = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

rac_data = {
    "GPT-4o": {
        "LONG":  [0.500, 0.625, 0.667, 0.562, 0.550, 0.500, 0.464, 0.406, 0.389, 0.350],
        "SHORT": [1.000, 1.000, 0.867, 0.850, 0.840, 0.833, 0.857, 0.875, 0.800, 0.740],
    },
    "Llama 3.1 70B": {
        "LONG":  [0.750, 0.875, 0.667, 0.688, 0.750, 0.708, 0.643, 0.562, 0.528, 0.475],
        "SHORT": [1.000, 1.000, 1.000, 0.900, 0.880, 0.833, 0.771, 0.750, 0.711, 0.660],
    },
    "Llama 3.1 8B": {
        "LONG":  [0.750, 0.750, 0.500, 0.562, 0.450, 0.417, 0.357, 0.344, 0.333, 0.300],
        "SHORT": [0.800, 0.400, 0.600, 0.500, 0.520, 0.467, 0.400, 0.375, 0.356, 0.320],
    },
    "o1": {
        "LONG":  [1.000, 0.875, 0.750, 0.750, 0.750, 0.708, 0.679, 0.688, 0.639, 0.575],
        "SHORT": [1.000, 0.900, 0.933, 0.900, 0.920, 0.933, 0.914, 0.925, 0.889, 0.840],
    },
}


# SE data: (SE_threshold, ACC, NUM) per model/type
# LONG total = 40, SHORT total = 50
se_data = {
    "GPT-4o": {
        "LONG": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [0.692, 0.692, 0.588, 0.444, 0.464, 0.419, 0.412, 0.368, 0.368, 0.368, 0.350],
            "num": [13, 13, 17, 27, 28, 31, 34, 38, 38, 38, 40],
        },
        "SHORT": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [0.853, 0.853, 0.872, 0.857, 0.837, 0.800, 0.755, 0.755, 0.755, 0.755, 0.740],
            "num": [34, 34, 39, 42, 43, 45, 49, 49, 49, 49, 50],
        },
    },
    "Llama 3.1 70B": {
        "LONG": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [0.857, 0.857, 0.778, 0.727, 0.643, 0.688, 0.739, 0.621, 0.581, 0.528, 0.475],
            "num": [7, 7, 9, 11, 14, 16, 23, 29, 31, 36, 40],
        },
        "SHORT": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [1.000, 1.000, 0.905, 0.846, 0.828, 0.771, 0.756, 0.721, 0.688, 0.660, 0.660],
            "num": [12, 12, 21, 26, 29, 35, 41, 43, 48, 50, 50],
        },
    },
    "Llama 3.1 8B": {
        "LONG": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [None, None, 1.000, 0.750, 0.750, 0.667, 0.545, 0.562, 0.500, 0.400, 0.300],
            "num": [0, 0, 2, 4, 4, 6, 11, 16, 18, 25, 40],
        },
        "SHORT": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [0.750, 0.750, 0.571, 0.600, 0.588, 0.520, 0.519, 0.389, 0.385, 0.357, 0.320],
            "num": [4, 4, 7, 15, 17, 25, 27, 36, 39, 42, 50],
        },
    },
    "o1": {
        "LONG": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [0.737, 0.737, 0.682, 0.690, 0.647, 0.629, 0.590, 0.575, 0.575, 0.575, 0.575],
            "num": [19, 19, 22, 29, 34, 35, 39, 40, 40, 40, 40],
        },
        "SHORT": {
            "se":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "acc": [0.941, 0.941, 0.927, 0.930, 0.909, 0.875, 0.875, 0.857, 0.857, 0.840, 0.840],
            "num": [34, 34, 41, 43, 44, 48, 48, 49, 49, 50, 50],
        },
    },
}

# Total number of questions per type
se_totals = {"LONG": 40, "SHORT": 50}


def save_figure(fig, name):
    """Save figure in PNG, SVG, and TIFF formats."""
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg")
    fig.savefig(base + ".tif", dpi=300)
    print(f"Saved {name} (.png, .svg, .tif)")


# =============================================================
# Figure 1: Accuracy Comparison (SHORT vs LONG)
# =============================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))

bar_width = 0.15
x_positions = range(len(models))
offsets = {"SHORT": -bar_width / 2, "LONG": bar_width / 2}

for mi, model in enumerate(models):
    for qtype in ("SHORT", "LONG"):
        ax1.bar(
            mi + offsets[qtype],
            accuracy[model][qtype],
            width=bar_width,
            color=color_map[model][qtype],
            edgecolor="none",
        )

ax1.set_xticks(list(x_positions))
ax1.set_xticklabels(models)
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.0)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.grid(axis="x", visible=False)
ax1.set_xlim(-0.5, len(models) - 0.5)

# Vertical separators
for x in [i + 0.5 for i in range(len(models) - 1)]:
    ax1.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.4, zorder=0)

# Legend: model colors
model_patches = [Patch(facecolor=color_map[m]["SHORT"], label=m) for m in models]
leg1 = ax1.legend(handles=model_patches, title="Models", loc="upper left", frameon=True)

# Legend: SHORT vs LONG
short_tuple = tuple(Patch(facecolor=color_map[m]["SHORT"]) for m in models)
long_tuple = tuple(Patch(facecolor=color_map[m]["LONG"]) for m in models)
leg2 = ax1.legend(
    [short_tuple, long_tuple],
    ["Short Answers", "Long Answers"],
    title="Question Type",
    loc="upper right",
    frameon=True,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    borderaxespad=1.0,
    labelspacing=0.8,
    handlelength=3.2,
    handleheight=1.0,
)
ax1.add_artist(leg1)

plt.tight_layout()
save_figure(fig1, "FigureSup1_Accuracy")
plt.close(fig1)


# =============================================================
# Figure 2: AUROC & AURAC Comparison (side-by-side subplots)
# =============================================================
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric_data, title in [(ax2a, auroc, "AUROC"), (ax2b, aurac, "AURAC")]:
    for mi, model in enumerate(models):
        for qtype in ("SHORT", "LONG"):
            ax.bar(
                mi + offsets[qtype],
                metric_data[model][qtype],
                width=bar_width,
                color=color_map[model][qtype],
                edgecolor="none",
            )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(models)
    ax.set_ylabel(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, len(models) - 0.5)

    for x in [i + 0.5 for i in range(len(models) - 1)]:
        ax.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.4, zorder=0)

# Shared legend on right subplot
model_patches = [Patch(facecolor=color_map[m]["SHORT"], label=m) for m in models]
leg1 = ax2b.legend(handles=model_patches, title="Models", loc="upper left", frameon=True)

short_tuple = tuple(Patch(facecolor=color_map[m]["SHORT"]) for m in models)
long_tuple = tuple(Patch(facecolor=color_map[m]["LONG"]) for m in models)
leg2 = ax2b.legend(
    [short_tuple, long_tuple],
    ["Short Answers", "Long Answers"],
    title="Question Type",
    loc="upper right",
    frameon=True,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    borderaxespad=1.0,
    labelspacing=0.8,
    handlelength=3.2,
    handleheight=1.0,
)
ax2b.add_artist(leg1)

plt.tight_layout()
save_figure(fig2, "FigureSup1_AUROC_AURAC")
plt.close(fig2)


# =============================================================
# Figure 3: Rejection Accuracy Curves (RAC) – one subplot per model
# =============================================================
fig3, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for ax, model in zip(axes.flat, models):
    ax.plot(
        rac_pct, rac_data[model]["SHORT"],
        color=color_map[model]["SHORT"], linewidth=2, marker="o",
        markersize=5, label="Short Answers",
    )
    ax.plot(
        rac_pct, rac_data[model]["LONG"],
        color=color_map[model]["LONG"], linewidth=2, marker="s",
        markersize=5, linestyle="--", label="Long Answers",
    )
    ax.set_title(model)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(5, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

# Shared axis labels
for ax in axes[1, :]:
    ax.set_xlabel("Fraction of Retained Questions (%)")
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy After Rejection")

# Shared legend from first subplot
handles, labels = axes[0, 0].get_legend_handles_labels()
fig3.legend(handles, labels, loc="lower center", ncol=2, frameon=True, fontsize=10)

plt.tight_layout()
fig3.subplots_adjust(bottom=0.13)
save_figure(fig3, "FigureSup1_RAC")
plt.close(fig3)

# =============================================================
# Figure 4: SE – Accuracy vs Fraction of Retained Questions
# =============================================================
fig4, axes4 = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

for ax, model in zip(axes4.flat, models):
    for qtype, ls, marker in [("SHORT", "-", "o"), ("LONG", "--", "s")]:
        d = se_data[model][qtype]
        total = se_totals[qtype]
        # Filter out None values and skip SE=0.1 (always identical to 0.0)
        frac = []
        acc_vals = []
        se_labels = []
        for se_val, a, n in zip(d["se"], d["acc"], d["num"]):
            if a is not None and se_val != 0.1:
                frac.append(n / total * 100)
                acc_vals.append(a)
                se_labels.append(se_val)

        ax.plot(
            frac, acc_vals,
            color=color_map[model][qtype], linewidth=2,
            marker=marker, markersize=5, linestyle=ls,
            label=f"{'Short' if qtype == 'SHORT' else 'Long'} Answers",
        )
        # Annotate each point with SE value
        # Group nearby points (< 4% apart) into clusters
        clusters = []
        current_cluster = [0]
        for idx in range(1, len(frac)):
            if abs(frac[idx] - frac[idx - 1]) < 4:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        # Pre-defined angle offsets (in points) for spreading clustered labels
        # Prefer upward, downward, and left directions (curve goes upper-right)
        spread_offsets = [
            (0, 16), (0, -16), (-14, 12), (-14, -12),
            (-16, 0), (-10, 16),
        ]

        # Manual overrides for specific (model, qtype, se_val) positions
        # None value means "skip this label entirely"
        manual_overrides = {
            ("Llama 3.1 8B", "LONG", 1.0): (0, -12),
            ("Llama 3.1 8B", "LONG", 0.0): (10, 12),
            ("Llama 3.1 8B", "LONG", 0.8): (0, -12),
            ("Llama 3.1 8B", "LONG", 0.6): (0, -12),
            ("Llama 3.1 8B", "LONG", 0.4): (-14, 0),
            ("Llama 3.1 8B", "LONG", 0.2): (-14, 0),
            ("Llama 3.1 8B", "SHORT", 0.0): (14, 0),
            ("Llama 3.1 8B", "SHORT", 1.0): None,
            ("o1", "SHORT", 1.0): (-10, 0),
            ("o1", "SHORT", 0.8): (10, -16),
            ("o1", "SHORT", 0.6): (10, -12),
            ("Llama 3.1 70B", "SHORT", 0.0): (0, -12),
            ("GPT-4o", "SHORT", 1.0): (-10, 0),
            ("o1", "LONG", 1.0): (0, 8),
            ("o1", "SHORT", 0.9): (0, -8),
        }

        ann_idx = 0  # running index for alternating
        for cluster in clusters:
            if len(cluster) == 1:
                # Isolated point: simple alternating above/below
                i = cluster[0]
                override = manual_overrides.get((model, qtype, se_labels[i]))
                if override is None and (model, qtype, se_labels[i]) in manual_overrides:
                    ann_idx += 1
                    continue
                if override:
                    dx, dy = override
                else:
                    dx, dy = 0, (9 if ann_idx % 2 == 0 else -12)
                use_arrow = override is not None and dx != 0
                ax.annotate(
                    f"{se_labels[i]:.1f}",
                    (frac[i], acc_vals[i]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=8, ha="center", color="black",
                    **(dict(arrowprops=dict(arrowstyle="-", color="gray", lw=0.6)) if use_arrow else {}),
                )
                ann_idx += 1
            else:
                # Clustered points: fan out with leader lines
                for j, i in enumerate(cluster):
                    override = manual_overrides.get((model, qtype, se_labels[i]))
                    if override is None and (model, qtype, se_labels[i]) in manual_overrides:
                        continue
                    if override:
                        dx, dy = override
                    else:
                        dx, dy = spread_offsets[j % len(spread_offsets)]
                    # Scale up offset for larger clusters
                    scale = 1.4 if len(cluster) > 2 else 1.0
                    ax.annotate(
                        f"{se_labels[i]:.1f}",
                        (frac[i], acc_vals[i]),
                        textcoords="offset points",
                        xytext=(dx * scale, dy * scale),
                        fontsize=8, ha="center", color="black",
                        arrowprops=dict(
                            arrowstyle="-",
                            color="gray",
                            lw=0.6,
                        ),
                    )
                ann_idx += len(cluster)

    ax.set_title(model)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(110, -5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

# Shared axis labels
for ax in axes4[1, :]:
    ax.set_xlabel("Fraction of Retained Questions (%)")
for ax in axes4[:, 0]:
    ax.set_ylabel("Accuracy After Rejection")

# Shared legend with neutral gray colors (applies to all subplots)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color="#404040", linewidth=2, marker="o", markersize=5, linestyle="-", label="Short Answers"),
    Line2D([0], [0], color="#B0B0B0", linewidth=2, marker="s", markersize=5, linestyle="--", label="Long Answers"),
]
fig4.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=True, fontsize=10)

plt.tight_layout()
fig4.subplots_adjust(bottom=0.13)
save_figure(fig4, "FigureSup1_SE")
plt.close(fig4)

print("All figures generated successfully.")
