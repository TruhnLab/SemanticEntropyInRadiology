# %%
# LogProb-based rejection analysis for GPT-4o
# Compares semantic entropy (SE) with token-level log-probability as uncertainty measure
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import sklearn.metrics
import tiktoken

_GPT4O_ENC = tiktoken.encoding_for_model("gpt-4o")

# ── paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "newFigure")

EVAL_FILES = [
    "EVAL_Image_Acquisition_GPT4o.csv",
    "EVAL_Guidelines_and_Indications_GPT4o.csv",
    "EVAL_Imaging_Education_GPT4o.csv",
    "EVAL_Research_GPT4o.csv",
]

CLUSTER_FILE = (
    "answerClustering Image Acquisition GPT4o_"
    " Guidelines and Indications GPT4o_"
    " Imaging Education GPT4o_"
    " Research GPT4o.csv"
)

# ── colors (consistent with createFigureSup1.py style) ──────────────────────
COLOR_SE = "#548235"        # green (GPT-4o color)
COLOR_RAW = "#2E75B6"       # blue
COLOR_NORM = "#BF9000"      # gold

METHOD_STYLE = {
    "SE":            {"color": COLOR_SE,   "marker": "o", "linestyle": "-"},
    "Raw LogProb":   {"color": COLOR_RAW,  "marker": "s", "linestyle": "--"},
    "Norm. LogProb": {"color": COLOR_NORM, "marker": "^", "linestyle": "-."},
}

# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Load and join data
# ═════════════════════════════════════════════════════════════════════════════

def load_eval_logprobs():
    """Load logProb of id=0 answers from GPT-4o EVAL files. Returns {prompt: (logProb, llmAnswer)}."""
    result = {}
    for fname in EVAL_FILES:
        path = os.path.join(CACHE_DIR, fname)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            idx_id = header.index("id")
            idx_prompt = header.index("prompt")
            idx_answer = header.index("llmAnswer")
            idx_lp = header.index("logProb")
            for row in reader:
                if row[idx_id] == "0":
                    lp = row[idx_lp]
                    if lp and lp != "None":
                        result[row[idx_prompt]] = (float(lp), row[idx_answer])
    return result


def load_clustering():
    """Load SE and correctness from answerClustering file. Returns {prompt: (SE, isCorrect)}."""
    result = {}
    path = os.path.join(CACHE_DIR, CLUSTER_FILE)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_se = header.index("discreteSemanticEntropy")
        idx_correct = header.index("isAnswerCorrect")
        idx_prompt = header.index("usedPrompt")
        for row in reader:
            se = float(row[idx_se])
            correct = row[idx_correct] == "True"
            result[row[idx_prompt]] = (se, correct)
    return result


def build_joined_data():
    """Join EVAL logProbs with clustering data. Returns list of dicts."""
    logprobs = load_eval_logprobs()
    clustering = load_clustering()

    joined = []
    for prompt, (se, correct) in clustering.items():
        if prompt in logprobs:
            lp, answer_text = logprobs[prompt]
            token_count = max(len(_GPT4O_ENC.encode(answer_text)), 1)
            joined.append({
                "se": se,
                "logprob_raw": lp,
                "logprob_norm": lp / token_count,
                "mean_prob": math.exp(lp / token_count),
                "correct": correct,
            })

    print(f"Joined {len(joined)} questions (of {len(clustering)} clustering / {len(logprobs)} EVAL entries)")
    return joined


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Metrics
# ═════════════════════════════════════════════════════════════════════════════

def calc_auroc(data, confidence_key):
    """AUROC via pairwise comparison (same method as clusterAnswers.py)."""
    true_conf = [d[confidence_key] for d in data if d["correct"]]
    false_conf = [d[confidence_key] for d in data if not d["correct"]]
    if not true_conf or not false_conf:
        return float("nan")
    comparisons = []
    for t in true_conf:
        for f_ in false_conf:
            comparisons.append(1 if t > f_ else 0)
    return sum(comparisons) / len(comparisons)


def calc_rac(data, confidence_key, cut_off_percent):
    """Rejection accuracy: keep top cut_off_percent by confidence."""
    sorted_data = sorted(data, key=lambda d: d[confidence_key], reverse=True)
    n_keep = max(round(cut_off_percent * len(sorted_data)), 1)
    subset = sorted_data[:n_keep]
    return sum(1 for d in subset if d["correct"]) / len(subset)


def calc_aurac(data, confidence_key):
    """AUC of the rejection accuracy curve (element-wise)."""
    sorted_data = sorted(data, key=lambda d: d[confidence_key], reverse=True)
    points = []
    for i in range(1, len(sorted_data) + 1):
        x = i / len(sorted_data)
        acc = sum(1 for d in sorted_data[:i] if d["correct"]) / i
        points.append((x, acc))
    xs, ys = zip(*points)
    return sklearn.metrics.auc(xs, ys)


def calc_threshold_curve(data, confidence_key, n_steps=20):
    """Accuracy as a function of confidence threshold.

    Returns (fractions_retained[], accuracies[], threshold_labels[]).
    Thresholds sweep from min to max of confidence_key.
    """
    values = [d[confidence_key] for d in data]
    lo, hi = min(values), max(values)
    thresholds = np.linspace(lo, hi, n_steps)
    total = len(data)
    fracs, accs, labels = [], [], []
    for thr in thresholds:
        subset = [d for d in data if d[confidence_key] >= thr]
        if not subset:
            continue
        frac = len(subset) / total * 100
        acc = sum(1 for d in subset if d["correct"]) / len(subset)
        fracs.append(frac)
        accs.append(acc)
        labels.append(thr)
    return fracs, accs, labels


def calc_se_threshold_curve(data, se_thresholds=None):
    """SE threshold curve (analogous to eval.py): keep questions with SE <= threshold."""
    if se_thresholds is None:
        se_thresholds = [i * 0.1 for i in range(11) if i != 1]  # skip 0.1 (always identical to 0.0)
    total = len(data)
    fracs, accs, labels = [], [], []
    for thr in se_thresholds:
        subset = [d for d in data if round(d["se"], 5) <= thr]
        if not subset:
            continue
        frac = len(subset) / total * 100
        acc = sum(1 for d in subset if d["correct"]) / len(subset)
        fracs.append(frac)
        accs.append(acc)
        labels.append(thr)
    return fracs, accs, labels


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Figures
# ═════════════════════════════════════════════════════════════════════════════

def save_figure(fig, name):
    base = os.path.join(FIG_DIR, name)
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg")
    fig.savefig(base + ".tif", dpi=300)
    print(f"Saved {name} (.png, .svg, .tif)")


# For SE: confidence = negative entropy (higher = more confident)
CONFIDENCE_KEYS = {
    "SE":            "neg_se",
    "Raw LogProb":   "logprob_raw",
    "Norm. LogProb": "logprob_norm",
}


def prepare_data(data):
    """Add neg_se field for SE-based confidence."""
    for d in data:
        d["neg_se"] = -d["se"]
    return data


# ── Figure 1: AUROC & AURAC bar chart ───────────────────────────────────────
def fig_auroc_aurac(data, methods=None, suffix=""):
    if methods is None:
        methods = list(METHOD_STYLE.keys())
    auroc_vals = [calc_auroc(data, CONFIDENCE_KEYS[m]) for m in methods]
    aurac_vals = [calc_aurac(data, CONFIDENCE_KEYS[m]) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    bar_width = 0.25
    x = np.arange(len(methods))

    for ax, vals, title in [(ax1, auroc_vals, "AUROC"), (ax2, aurac_vals, "AURAC")]:
        bars = ax.bar(
            x, vals, width=bar_width,
            color=[METHOD_STYLE[m]["color"] for m in methods],
            edgecolor="none",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, len(methods) - 0.5)
        # value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    plt.tight_layout()
    save_figure(fig, f"FigureLogProb_AUROC_AURAC{suffix}")
    plt.close(fig)
    return auroc_vals, aurac_vals


# ── Figure 2: RAC curves ────────────────────────────────────────────────────
def fig_rac(data, methods=None, suffix=""):
    if methods is None:
        methods = list(METHOD_STYLE.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    rac_pct = list(range(10, 110, 10))

    for method in methods:
        s = METHOD_STYLE[method]
        rac_vals = [calc_rac(data, CONFIDENCE_KEYS[method], p / 100) for p in rac_pct]
        ax.plot(
            rac_pct, rac_vals,
            color=s["color"], linewidth=2,
            marker=s["marker"], markersize=5,
            linestyle=s["linestyle"], label=method,
        )

    ax.set_xlabel("Fraction of Retained Questions (%)")
    ax.set_ylabel("Accuracy After Rejection")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(5, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    save_figure(fig, f"FigureLogProb_RAC{suffix}")
    plt.close(fig)


# ── Figure 3: Scatter SE vs LogProb ─────────────────────────────────────────
def fig_scatter(data, suffix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    se_vals = [d["se"] for d in data]
    raw_lp = [d["logprob_raw"] for d in data]
    norm_lp = [d["logprob_norm"] for d in data]
    colors = ["#548235" if d["correct"] else "#C55A11" for d in data]

    for ax, lp_vals, label in [(ax1, raw_lp, "Raw LogProb"), (ax2, norm_lp, "Norm. LogProb")]:
        ax.scatter(se_vals, lp_vals, c=colors, s=40, alpha=0.7, edgecolors="none")
        ax.set_xlabel("Semantic Entropy")
        ax.set_ylabel(label)
        ax.grid(axis="both", linestyle="--", alpha=0.5)

        # Pearson correlation
        r = np.corrcoef(se_vals, lp_vals)[0, 1]
        ax.annotate(
            f"r = {r:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=10, va="top",
        )

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#548235",
               markersize=8, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C55A11",
               markersize=8, label="Incorrect"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=True, fontsize=10)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    save_figure(fig, f"FigureLogProb_Scatter{suffix}")
    plt.close(fig)


# ── Figure 4: Threshold accuracy curves ─────────────────────────────────────

def _annotate_points(ax, fracs, accs, labels, fmt=".1f", side="above",
                     reverse=False):
    """Annotate points with threshold labels, skipping crowded points.

    Iterates through points and only labels a point if no already-labeled
    point is within 4% (in x). side controls above/below placement.
    reverse=True iterates from last to first (useful for SE where the
    rightmost point at 100% retention should get priority).
    """
    if not fracs:
        return

    dy = 10 if side == "above" else -12
    labeled_fracs = []

    indices = range(len(fracs) - 1, -1, -1) if reverse else range(len(fracs))
    for i in indices:
        # Skip if any already-labeled point is within 4%
        if any(abs(fracs[i] - lf) < 4 for lf in labeled_fracs):
            continue
        ax.annotate(
            f"{labels[i]:{fmt}}", (fracs[i], accs[i]),
            textcoords="offset points", xytext=(0, dy),
            fontsize=8, ha="center", color="black",
        )
        labeled_fracs.append(fracs[i])


def fig_threshold(data, methods=None, suffix=""):
    if methods is None:
        methods = list(METHOD_STYLE.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        if method == "SE":
            fracs, accs, labels = calc_se_threshold_curve(data)
            fmt = ".1f"
        else:
            key = {"Raw LogProb": "logprob_raw", "Norm. LogProb": "logprob_norm"}[method]
            fracs, accs, labels = calc_threshold_curve(data, key, n_steps=20)
            fmt = ".2f"
            labels = [abs(l) if abs(l) < 0.005 else l for l in labels]  # avoid -0.00
        s = METHOD_STYLE[method]
        ax.plot(
            fracs, accs,
            color=s["color"], linewidth=2, marker=s["marker"],
            markersize=5, linestyle=s["linestyle"], label=method,
        )
        side = "above" if method == "SE" else "below"
        rev = method == "SE"  # SE: start from 1.0 (rightmost) so it gets priority
        _annotate_points(ax, fracs, accs, labels, fmt=fmt, side=side, reverse=rev)

    ax.set_xlabel("Fraction of Retained Questions (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(110, -5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    plt.tight_layout()
    save_figure(fig, f"FigureLogProb_Threshold{suffix}")
    plt.close(fig)


# ── Figure 5: SE vs Mean Token Probability ──────────────────────────────────
def fig_threshold_prob(data, suffix=""):
    """Like FigureLogProb_Threshold_noRaw but with mean token probability
    (exp of norm logprob) instead of log prob on the yellow curve."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # SE curve
    fracs, accs, labels = calc_se_threshold_curve(data)
    s = METHOD_STYLE["SE"]
    ax.plot(
        fracs, accs,
        color=s["color"], linewidth=2, marker=s["marker"],
        markersize=5, linestyle=s["linestyle"], label="SE",
    )
    _annotate_points(ax, fracs, accs, labels, fmt=".1f", side="above", reverse=True)

    # Mean token probability curve
    fracs_p, accs_p, labels_p = calc_threshold_curve(data, "mean_prob", n_steps=20)
    s = METHOD_STYLE["Norm. LogProb"]  # same gold style
    ax.plot(
        fracs_p, accs_p,
        color=s["color"], linewidth=2, marker=s["marker"],
        markersize=5, linestyle=s["linestyle"], label="Mean Token Prob.",
    )
    labels_p = [abs(l) if abs(l) < 0.005 else l for l in labels_p]
    _annotate_points(ax, fracs_p, accs_p, labels_p, fmt=".2f", side="below", reverse=False)

    ax.set_xlabel("Fraction of Retained Questions (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(110, -5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.legend(loc="lower left", frameon=True, fontsize=10)

    plt.tight_layout()
    save_figure(fig, f"FigureLogProb_ThresholdProb{suffix}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Console output + main
# ═════════════════════════════════════════════════════════════════════════════

def print_metrics(data):
    methods = list(METHOD_STYLE.keys())
    print("\n" + "=" * 60)
    print("LogProb vs Semantic Entropy — GPT-4o (all questions)")
    print("=" * 60)
    print(f"Number of questions: {len(data)}")
    print(f"Accuracy: {sum(1 for d in data if d['correct']) / len(data):.3f}")
    print()

    for method in methods:
        key = CONFIDENCE_KEYS[method]
        auroc = calc_auroc(data, key)
        aurac = calc_aurac(data, key)
        print(f"--- {method} ---")
        print(f"  AUROC: {auroc:.3f}")
        print(f"  AURAC: {aurac:.3f}")
        print(f"  [RAC = rejection accuracy]")
        for p in range(10, 110, 10):
            rac = calc_rac(data, key, p / 100)
            print(f"  RAC at {p:3d}%: {rac:.3f}")
        print()


def main():
    data = build_joined_data()
    data = prepare_data(data)
    print_metrics(data)

    print("\nGenerating figures (all 3 methods)...")
    fig_auroc_aurac(data)
    fig_rac(data)
    fig_scatter(data)
    fig_threshold(data)

    # Variant without Raw LogProb (SE vs Norm. LogProb only)
    norm_only = ["SE", "Norm. LogProb"]
    print("\nGenerating figures (SE vs Norm. LogProb only)...")
    fig_auroc_aurac(data, methods=norm_only, suffix="_noRaw")
    fig_rac(data, methods=norm_only, suffix="_noRaw")
    fig_threshold(data, methods=norm_only, suffix="_noRaw")

    print("\nGenerating SE vs Mean Token Probability figure...")
    fig_threshold_prob(data)
    print("\nDone.")


if __name__ == "__main__":
    main()
