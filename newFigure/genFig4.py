# -*- coding: utf-8 -*-
# Scatterplot: Absolute Accuracy Gain vs. Remaining Questions (ENT=0.5 vs ENT=1.0)
# Farben = Modelle, Marker = Kategorien

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# -----------------------------
# Daten (aus deinem Prompt)
# -----------------------------
rows = [#corrected
    # o1
    {"model": "o1", "category": "Image Acquisition",         "acc_ent05": 0.842, "acc_ent10": 0.800, "num_ent05": 19, "num_ent10": 20},
    {"model": "o1", "category": "Guidelines and Indications","acc_ent05": 0.591, "acc_ent10": 0.520, "num_ent05": 22, "num_ent10": 25},
    {"model": "o1", "category": "Imaging Education",         "acc_ent05": 0.875, "acc_ent10": 0.880, "num_ent05": 24, "num_ent10": 25},
    {"model": "o1", "category": "Research",                  "acc_ent05": 0.778, "acc_ent10": 0.700, "num_ent05": 18, "num_ent10": 20},

    # GPT-4o
    {"model": "GPT-4o", "category": "Image Acquisition",         "acc_ent05": 0.778, "acc_ent10": 0.750, "num_ent05": 18, "num_ent10": 20},
    {"model": "GPT-4o", "category": "Guidelines and Indications","acc_ent05": 0.526, "acc_ent10": 0.440, "num_ent05": 19, "num_ent10": 25},
    {"model": "GPT-4o", "category": "Imaging Education",         "acc_ent05": 0.818, "acc_ent10": 0.720, "num_ent05": 22, "num_ent10": 25},
    {"model": "GPT-4o", "category": "Research",                  "acc_ent05": 0.765, "acc_ent10": 0.650, "num_ent05": 17, "num_ent10": 20},

    # LLaMA 3.1 70B
    {"model": "LLaMA 3.1 70B", "category": "Image Acquisition",         "acc_ent05": 0.818, "acc_ent10": 0.650, "num_ent05": 11, "num_ent10": 20},
    {"model": "LLaMA 3.1 70B", "category": "Guidelines and Indications","acc_ent05": 0.700, "acc_ent10": 0.440, "num_ent05": 10, "num_ent10": 25},
    {"model": "LLaMA 3.1 70B", "category": "Imaging Education",         "acc_ent05": 0.684, "acc_ent10": 0.560, "num_ent05": 19, "num_ent10": 25},
    {"model": "LLaMA 3.1 70B", "category": "Research",                  "acc_ent05": 0.818, "acc_ent10": 0.700, "num_ent05": 11, "num_ent10": 20},

    # LLaMA 3.1 8B
    {"model": "LLaMA 3.1 8B", "category": "Image Acquisition",         "acc_ent05": 0.750, "acc_ent10": 0.350, "num_ent05": 4,  "num_ent10": 20},
    {"model": "LLaMA 3.1 8B", "category": "Guidelines and Indications","acc_ent05": 0.600, "acc_ent10": 0.200, "num_ent05": 5,  "num_ent10": 25},
    {"model": "LLaMA 3.1 8B", "category": "Imaging Education",         "acc_ent05": 0.222, "acc_ent10": 0.160, "num_ent05": 9,  "num_ent10": 25},
    {"model": "LLaMA 3.1 8B", "category": "Research",                  "acc_ent05": 0.692, "acc_ent10": 0.600, "num_ent05": 13, "num_ent10": 20},
]

df = pd.DataFrame(rows)

# -----------------------------
# Kennzahlen berechnen
# -----------------------------
df["abs_gain"] = df["acc_ent05"] - df["acc_ent10"]
df["remaining_pct"] = 100.0 * df["num_ent05"] / df["num_ent10"]

# -----------------------------
# Farben & Marker
# -----------------------------
color_map = {
    "GPT-4o": "#548235",
    "LLaMA 3.1 70B": "#BF9000",
    "LLaMA 3.1 8B": "#C55A11",
    "o1": "#2E75B6",
}

marker_map = {
    "Image Acquisition": "^",           # triangle
    "Guidelines and Indications": "s",  # square
    "Imaging Education": "o",           # circle
    "Research": "x",                    # x
}

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(8, 6))

# Manuelle Offsets für überlappende Punkte (LLaMA 3.1 8B: Image Acquisition
# und Guidelines and Indications liegen beide exakt bei (0.40, 20%)).
# Symbole werden leicht versetzt dargestellt, ein kleiner Pfeil verweist
# auf die tatsächliche Datenposition.
overlap_offsets = {
    ("LLaMA 3.1 8B", "Image Acquisition"):          ( 0.020,  4.5),  # nach oben rechts
    ("LLaMA 3.1 8B", "Guidelines and Indications"): (-0.020,  4.5),  # nach oben links
}

# Punkte: groß (s=120), Farbe nach Modell, Marker nach Kategorie
for _, r in df.iterrows():
    x, y = r["abs_gain"], r["remaining_pct"]
    key = (r["model"], r["category"])
    if key in overlap_offsets:
        dx, dy = overlap_offsets[key]
        x_disp, y_disp = x + dx, y + dy
        plt.annotate(
            "",
            xy=(x, y),
            xytext=(x_disp, y_disp),
            arrowprops=dict(
                arrowstyle="->",
                color=color_map[r["model"]],
                lw=1.0,
                shrinkA=6,
                shrinkB=2,
            ),
        )
        plt.scatter(
            x_disp, y_disp,
            c=color_map[r["model"]],
            marker=marker_map[r["category"]],
            s=120,
            alpha=0.9,
            edgecolors="none",
        )
    else:
        plt.scatter(
            x, y,
            c=color_map[r["model"]],
            marker=marker_map[r["category"]],
            s=120,
            alpha=0.9,
            edgecolors="none",
        )

# Achsen
plt.xlabel("Absolute Accuracy Gain (Reject SE > 0.5 vs All)")
plt.ylabel("Remaining Questions when Rejecting at SE > 0.5 (%)")
plt.xlim(-0.1, 0.5)
plt.ylim(0, 100)

# Ticks (Labels) alle 0.1 / 20%, feines Minor-Grid alle 0.05 / 5%
ax = plt.gca()
ax.set_xticks(np.round(np.arange(-0.10, 0.50 + 0.001, 0.10), 2))   # labeled
ax.set_yticks(np.arange(0, 100 + 0.001, 20))                       # labeled
ax.set_xticks(np.round(np.arange(-0.10, 0.50 + 0.001, 0.05), 2), minor=True)
ax.set_yticks(np.arange(0, 100 + 0.001, 5), minor=True)

# Grid
ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.8)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.6)

# Vertikale Referenzlinie bei x = 0
ax.axvline(0, linestyle="-", linewidth=1)

# -----------------------------
# Kombinierte Legende (im Plot, oben rechts)
# -----------------------------
cat_handles = [
    Line2D([0], [0], marker=marker_map["Image Acquisition"], linestyle="None",
           markersize=9, markeredgewidth=1.5, color="black", label="Image Acquisition"),
    Line2D([0], [0], marker=marker_map["Guidelines and Indications"], linestyle="None",
           markersize=9, markeredgewidth=1.5, color="black", label="Guidelines and Indications"),
    Line2D([0], [0], marker=marker_map["Imaging Education"], linestyle="None",
           markersize=9, markeredgewidth=1.5, color="black", label="Imaging Education"),
    Line2D([0], [0], marker=marker_map["Research"], linestyle="None",
           markersize=10, markeredgewidth=1.8, color="black", label="Research"),
]
model_handles = [
    Line2D([0], [0], marker="o", linestyle="None", color=color_map["GPT-4o"], label="GPT-4o", markersize=9),
    Line2D([0], [0], marker="o", linestyle="None", color=color_map["LLaMA 3.1 70B"], label="Llama 3.1 70B", markersize=9),
    Line2D([0], [0], marker="o", linestyle="None", color=color_map["LLaMA 3.1 8B"], label="Llama 3.1 8B", markersize=9),
    Line2D([0], [0], marker="o", linestyle="None", color=color_map["o1"], label="o1", markersize=9),
]

# Optionaler visueller Trenner (leerer Handle)
sep = Line2D([], [], linestyle="None", label="")

legend_handles = [
    Line2D([], [], linestyle="None", label="Categories"),
    *cat_handles,
    sep,
    Line2D([], [], linestyle="None", label="Models"),
    *model_handles,
]

ax.legend(
    handles=legend_handles,
    loc="upper right",     # in der Achse, oben rechts
    framealpha=0.95,
    borderaxespad=0.6,
)

plt.tight_layout()

# Speichern (Dateinamen nach Bedarf anpassen)
plt.savefig("ent05_vs_ent10_scatter_final.png", dpi=150)
plt.savefig("ent05_vs_ent10_scatter_final.svg")
plt.savefig("ent05_vs_ent10_scatter_final_300dpi.tif", dpi=300)
plt.show()
