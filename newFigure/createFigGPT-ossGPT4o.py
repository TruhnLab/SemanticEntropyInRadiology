# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

out_dir = os.path.dirname(os.path.abspath(__file__))

# Total questions
total = 90

# --- Data ---
se_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

gpt_oss = {
    "acc": [0.84, 0.87, 0.83, 0.80, 0.81, 0.76, 0.74, 0.68, 0.66, 0.63],
    "num": [25, 30, 40, 45, 58, 66, 72, 80, 83, 90],
}

gpt4o = {
    "acc": [0.83, 0.82, 0.77, 0.76, 0.72, 0.69, 0.66, 0.66, 0.66, 0.63],
    "num": [47, 56, 69, 71, 76, 83, 87, 87, 87, 90],
}

# Fraction of retained questions (%)
gpt_oss["frac"] = [n / total * 100 for n in gpt_oss["num"]]
gpt4o["frac"] = [n / total * 100 for n in gpt4o["num"]]

# Colors (GPT-4o same as in createFigureSup1.py)
color_gpt4o = "#548235"
color_gpt_oss = "#7030A0"

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    gpt4o["frac"], gpt4o["acc"],
    color=color_gpt4o, linewidth=2, marker="o", markersize=5,
    linestyle="-", label="GPT-4o",
)
ax.plot(
    gpt_oss["frac"], gpt_oss["acc"],
    color=color_gpt_oss, linewidth=2, marker="s", markersize=5,
    linestyle="--", label="gpt-oss-120b",
)

# Annotate SE values: GPT-4o upper-left, GPT-oss lower-right
# Manual overrides: (model_key, se_val) -> (dx, dy)
manual_overrides = {
    ("gpt_oss", 0.2): (0, 12),
    ("gpt4o", 0.4): (-14, 0),
    ("gpt4o", 0.7): (0, 16),
    ("gpt4o", 0.8): (-14, 14),
    ("gpt4o", 0.9): (-20, 0),
}
# Which overrides get arrow lines
arrow_keys = {("gpt4o", 0.7), ("gpt4o", 0.8), ("gpt4o", 0.9)}

for data, default_offset, key in [(gpt4o, (-10, 8), "gpt4o"), (gpt_oss, (10, -10), "gpt_oss")]:
    for i, se in enumerate(se_vals):
        override = manual_overrides.get((key, se))
        dx, dy = override if override else default_offset
        use_arrow = (key, se) in arrow_keys
        ax.annotate(
            f"{se:.1f}",
            (data["frac"][i], data["acc"][i]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8, ha="center", color="black",
            **(dict(arrowprops=dict(arrowstyle="-", color="gray", lw=0.6)) if use_arrow else {}),
        )

ax.set_xlabel("Fraction of Retained Questions (%)")
ax.set_ylabel("Accuracy After Rejection")
ax.set_ylim(0.5, 1.0)
ax.set_xlim(105, 20)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.grid(axis="x", linestyle=":", alpha=0.3)
ax.legend(loc="lower left", frameon=True, fontsize=10)

plt.tight_layout()

# Save
for ext in (".png", ".svg", ".tif"):
    fig.savefig(os.path.join(out_dir, f"FigGPT-ossGPT4o{ext}"), dpi=300)
print("Saved FigGPT-ossGPT4o (.png, .svg, .tif)")
plt.close(fig)
