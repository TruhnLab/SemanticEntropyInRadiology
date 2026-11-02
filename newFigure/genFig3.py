# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

# -----------------------------
# Daten (Accuracy bei ENT 0.5 und 1.0)
# -----------------------------
models = ["GPT-4o", "LLaMA 3.1 70B", "LLaMA 3.1 8B", "o1"]
categories = ["All", "Guidelines & Indications", "Image Acquisition", "Imaging Education", "Research"]

data = { #corrected
    "All": {
        "GPT-4o": {0.5: 0.724, 1.0: 0.633},
        "LLaMA 3.1 70B": {0.5: 0.745, 1.0: 0.578},
        "LLaMA 3.1 8B": {0.5: 0.548, 1.0: 0.311},
        "o1": {0.5: 0.771, 1.0: 0.722},
    },
    "Guidelines & Indications": {
        "GPT-4o": {0.5: 0.526, 1.0: 0.440},
        "LLaMA 3.1 70B": {0.5: 0.700, 1.0: 0.440},
        "LLaMA 3.1 8B": {0.5: 0.600, 1.0: 0.200},
        "o1": {0.5: 0.591, 1.0: 0.520},
    },
    "Image Acquisition": {
        "GPT-4o": {0.5: 0.778, 1.0: 0.750},
        "LLaMA 3.1 70B": {0.5: 0.818, 1.0: 0.650},
        "LLaMA 3.1 8B": {0.5: 0.750, 1.0: 0.350},
        "o1": {0.5: 0.842, 1.0: 0.800},
    },
    "Imaging Education": {
        "GPT-4o": {0.5: 0.818, 1.0: 0.720},
        "LLaMA 3.1 70B": {0.5: 0.684, 1.0: 0.560},
        "LLaMA 3.1 8B": {0.5: 0.222, 1.0: 0.160},
        "o1": {0.5: 0.875, 1.0: 0.880},
    },
    "Research": {
        "GPT-4o": {0.5: 0.765, 1.0: 0.650},
        "LLaMA 3.1 70B": {0.5: 0.818, 1.0: 0.700},
        "LLaMA 3.1 8B": {0.5: 0.692, 1.0: 0.600},
        "o1": {0.5: 0.778, 1.0: 0.700},
    },
}

# -----------------------------
# Farben (beide Sättigungen im selben Dict)
# -----------------------------
color_map = {
    "GPT-4o": {0.5: "#548235", 1.0: "#C5E0B4"},
    "LLaMA 3.1 70B": {0.5: "#BF9000", 1.0: "#FFE699"},
    "LLaMA 3.1 8B": {0.5: "#C55A11", 1.0: "#F8CBAD"},
    "o1": {0.5: "#2E75B6", 1.0: "#BDD7EE"},
}

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

n_models = len(models)
bar_width = 0.08
group_centers = list(range(len(categories)))

# Modell-Offsets (vier Cluster je Kategorie)
cluster_step = 0.24  # Abstand zwischen Modell-Clustern innerhalb einer Kategorie
model_offsets = [((i - (n_models - 1) / 2) * cluster_step) for i in range(n_models)]
# Innerhalb des Modell-Clusters zwei Balken (ENT 0.5 links, ENT 1.0 rechts)
ent_offsets = {0.5: -bar_width / 2, 1.0: bar_width / 2}

for ci, cat in enumerate(categories):
    center = group_centers[ci]
    for mi, model in enumerate(models):
        x_base = center + model_offsets[mi]
        for ent in (0.5, 1.0):
            ax.bar(
                x_base + ent_offsets[ent],
                data[cat][model][ent],
                width=bar_width,
                color=color_map[model][ent],
                edgecolor="none",
            )

# Achsen/Styling an das Beispiel anlehnen
ax.set_xticks(group_centers)
ax.set_xticklabels(categories, rotation=0)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.grid(axis="x", visible=False)

# vertikale Trenner zwischen Kategorien (bei -0.5, 0.5, 1.5, ...)
ax.set_xlim(-0.5, len(categories) - 0.5)
for x in [-0.5] + [i + 0.5 for i in range(len(categories) - 1)] + [len(categories) - 0.5]:
    ax.axvline(x, linestyle="--", linewidth=0.8, color="gray", alpha=0.4, zorder=0)

# Legende links: Modelle (mit ENT 0.5 Farben)
model_patches = [Patch(facecolor=color_map[m][0.5], label=m.replace("LLaMA","Llama")) for m in models]
leg1 = ax.legend(handles=model_patches, title="Models", loc="upper left", frameon=True)

# Legende rechts: "Answered Questions" mit zwei Einträgen (All vs Reject SE > 0.5),
# jeder Eintrag ist ein Tuple aus 4 Patches in den jeweiligen Modellfarben.
all_tuple = tuple(Patch(facecolor=color_map[m][1.0]) for m in models)          # ENT 1.0 (All)
reject_tuple = tuple(Patch(facecolor=color_map[m][0.5]) for m in models)       # ENT 0.5 (Reject)

leg2 = ax.legend(
    [all_tuple, reject_tuple],
    ["All", "Reject SE > 0.5"],
    title="Answered Questions",
    loc="upper right",
    frameon=True,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    borderaxespad=1.0,
    labelspacing=0.8,
    handlelength=3.2,
    handleheight=1.0,
)
ax.add_artist(leg1)

plt.tight_layout()
out_path = "Figure3v2.png"
plt.savefig(out_path, dpi=300)
plt.savefig(out_path.replace(".png",".svg"))
plt.savefig(out_path.replace(".png",".tif"),dpi=300)
plt.show()
