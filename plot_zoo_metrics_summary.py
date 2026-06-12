"""
plot_zoo_metrics_summary.py — ZOO per-metric bar chart.

For a single dataset and threat model (default: cifar10, targeted), this reads each
optimizer's results.json and plots six panels: MAE, MSE, PSNR, SSIM, L-inf and L2
distortion, one coloured bar per optimizer.

Folder structure:
    <results-dir>/<dataset>/<targeted|untargeted>/<solver>/results.json
                                                          /original_*.png
                                                          /adversarial_*.png

MAE / MSE / PSNR / SSIM are read from results.json. L2 / L-inf are read from a
`distortion` block if present (PGD-style results.json); otherwise they are computed
from the original/adversarial image pairs in the same folder (ZOO-style results.json,
which does not store them). Any solver folder whose name contains "pgd" is skipped.

Usage
-----
    python plot_zoo_metrics_summary.py                       # cifar10, targeted
    python plot_zoo_metrics_summary.py --dataset mnist
    python plot_zoo_metrics_summary.py --attack-type untargeted
    python plot_zoo_metrics_summary.py --results-dir . --output zoo_metrics_cifar10.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


# ZOO optimizers, in the order used by the report figure.
DEFAULT_SOLVERS = ["adam", "newton", "sgd", "sgdsign", "signum", "lion", "adahessian"]

# (label, json-key, higher_is_better)
METRICS = [
    ("MAE",            "mae",  False),
    ("MSE",            "mse",  False),
    ("PSNR",           "psnr", True),
    ("SSIM",           "ssim", True),
    ("L-inf Distortion", "linf", False),
    ("L2 Distortion",  "l2",   False),
]


def configure_font():
    candidates = [
        "/mnt/c/Windows/Fonts/times.ttf",
        "/mnt/c/Windows/Fonts/timesbd.ttf",
        "/mnt/c/Windows/Fonts/timesi.ttf",
        "/mnt/c/Windows/Fonts/timesbi.ttf",
    ]
    found = False
    for path in candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                found = True
            except Exception:
                pass
    plt.rcParams["font.family"] = "Times New Roman" if found else "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]


def _mean(block):
    """Pull the mean from a metric block, tolerating the 'mean:' typo key."""
    if not isinstance(block, dict):
        return np.nan
    if "mean" in block:
        return float(block["mean"])
    if "mean:" in block:
        return float(block["mean:"])
    return np.nan


def distortion_from_images(folder):
    """Mean per-image L2 and L-inf over original/adversarial PNG pairs (pixels in [0, 1])."""
    n = len(glob.glob(os.path.join(folder, "original_*.png")))
    if n == 0:
        return np.nan, np.nan
    from PIL import Image  # local import so the script runs even without images present
    l2s, linfs = [], []
    for i in range(n):
        op = os.path.join(folder, "original_%d.png" % i)
        ap = os.path.join(folder, "adversarial_%d.png" % i)
        if not (os.path.exists(op) and os.path.exists(ap)):
            continue
        o = np.asarray(Image.open(op).convert("RGB"), dtype=np.float64) / 255.0
        a = np.asarray(Image.open(ap).convert("RGB"), dtype=np.float64) / 255.0
        d = (a - o).ravel()
        l2s.append(float(np.linalg.norm(d)))
        linfs.append(float(np.abs(d).max()))
    if not l2s:
        return np.nan, np.nan
    return float(np.mean(l2s)), float(np.mean(linfs))


def load_solver_metrics(folder):
    """Return {mae, mse, psnr, ssim, l2, linf} for one solver folder."""
    path = os.path.join(folder, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)

    m = {
        "mae":  _mean(r.get("mae")),
        "mse":  _mean(r.get("mse")),
        "psnr": _mean(r.get("psnr")),
        "ssim": _mean(r.get("ssim")),
    }

    # L2 / L-inf: prefer a stored distortion block (PGD-style), else derive from images.
    dist = r.get("distortion", {})
    l2 = dist.get("mean_l2_on_success")
    linf = dist.get("mean_linf_on_success")
    if l2 is None or linf is None:
        img_l2, img_linf = distortion_from_images(folder)
        l2 = img_l2 if l2 is None else l2
        linf = img_linf if linf is None else linf
    m["l2"] = float(l2) if l2 is not None else np.nan
    m["linf"] = float(linf) if linf is not None else np.nan
    return m


def main():
    p = argparse.ArgumentParser(description="ZOO per-metric bar chart (report Fig. 5).")
    p.add_argument("--results-dir", default=".",
                   help="Directory containing the <dataset>/ results tree (default: .)")
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "imagenet"],
                   help="Dataset to plot (default: cifar10)")
    p.add_argument("--attack-type", default="targeted", choices=["targeted", "untargeted"],
                   help="Threat model to plot (default: targeted)")
    p.add_argument("--exclude-solvers", nargs="*", default=[],
                   help="Optional solver names to exclude")
    p.add_argument("--output", default=None,
                   help="Output filename under plots/ (default: zoo_metrics_<dataset>_<attack>.png)")
    args = p.parse_args()

    configure_font()
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
    })

    exclude = {s.lower() for s in args.exclude_solvers}
    base = os.path.join(args.results_dir, args.dataset, args.attack_type)

    solvers, data = [], {}
    for solver in DEFAULT_SOLVERS:
        if solver in exclude or "pgd" in solver:   # skip pgd-named folders
            continue
        folder = os.path.join(base, solver)
        metrics = load_solver_metrics(folder)
        if metrics is None:
            print("skip (no results.json): %s" % folder)
            continue
        solvers.append(solver)
        data[solver] = metrics

    if not solvers:
        raise SystemExit("No ZOO solver results found under %s" % base)

    # Colour scheme matching the companion NES summary plot.
    cmap = plt.get_cmap("viridis_r")
    positions = np.linspace(0.2, 0.9, len(solvers))
    solver_colors = {s: cmap(pos) for s, pos in zip(solvers, positions)}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.05 * len(METRICS), 3.4), dpi=170)

    for ax, (label, key, higher) in zip(axes, METRICS):
        vals = [data[s][key] for s in solvers]
        x = np.arange(len(solvers))
        for xi, s, v in zip(x, solvers, vals):
            ax.bar(xi, v, color=solver_colors[s], edgecolor="black", linewidth=0.4, width=0.8)
        arrow = " \u2191" if higher else " \u2193"
        ax.set_title(label + arrow)
        ax.set_xticks([])
        ax.margins(y=0.12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=solver_colors[s],
               markeredgecolor="black", markeredgewidth=0.5, markersize=9, label=s)
        for s in solvers
    ]
    fig.legend(handles=handles, labels=solvers, loc="lower center",
               ncol=len(solvers), frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("%s attack on %s (ZOO) — distortion & quality metrics per optimizer"
                 % (args.attack_type.capitalize(), args.dataset.upper()), y=1.02)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    os.makedirs("plots", exist_ok=True)
    out = args.output or ("zoo_metrics_%s_%s.png" % (args.dataset, args.attack_type))
    out_path = os.path.join("plots", out)
    plt.savefig(out_path, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
