import argparse
import json
import os
 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
 
 
DEFAULT_SOLVERS = [
    "momentum", "nesterov", "adagrad", "adam", "sgd",
    "sgdsign", "signum", "lion", "newton", "adahessian",
]
DATASETS = ["mnist", "cifar10"]
 
 
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
 
    if found:
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
 
 
def load_metrics(results_dir, attack_type, dataset, solver):
    path = os.path.join(results_dir, dataset, attack_type, solver, "results.json")
    if not os.path.exists(path):
        return np.nan, np.nan
 
    with open(path, "r") as f:
        result = json.load(f)
 
    l2 = float(result.get("total_distortion", np.nan))
    q = float(result.get("queries", {}).get("mean_on_success", np.nan))
    return l2, q
 
 
def collect_pairs(results_dir, solvers):
    rows = []
    for dataset in DATASETS:
        for solver in solvers:
            u_l2, u_q = load_metrics(results_dir, "untargeted", dataset, solver)
            t_l2, t_q = load_metrics(results_dir, "targeted", dataset, solver)
            rows.append(
                {
                    "dataset": dataset,
                    "solver": solver,
                    "u_l2": u_l2,
                    "u_q": u_q,
                    "t_l2": t_l2,
                    "t_q": t_q,
                }
            )
    return rows
 
 
def valid(v):
    return not np.isnan(v)
 
 
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Targeted vs untargeted summary: diagonal comparison for L2 and queries"
        )
    )
    parser.add_argument(
        "--results-dir",
        default="nes_results",
        help="Root directory holding <dataset>/<targeted|untargeted>/<solver>/results.json",
    )
    parser.add_argument(
        "--exclude-solvers",
        nargs="*",
        default=[],
        help="Optional solver names to exclude from the plot",
    )
    parser.add_argument(
        "--output",
        default="targeted_vs_untargeted_easy_summary.png",
        help="Output filename under plots/",
    )
    args = parser.parse_args()
 
    configure_font()
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
    })
 
    exclude = {s.lower() for s in args.exclude_solvers}
    solvers = [s for s in DEFAULT_SOLVERS if s not in exclude]
    if not solvers:
        raise ValueError("All solvers were excluded; nothing to plot.")
 
    rows = collect_pairs(args.results_dir, solvers)
 
    os.makedirs("plots", exist_ok=True)
 
    # Color scheme requested in prior plots.
    cmap = plt.get_cmap("viridis_r")
    positions = np.linspace(0.2, 0.9, len(solvers))
    solver_colors = {solver: cmap(pos) for solver, pos in zip(solvers, positions)}
    solver_index = {solver: i for i, solver in enumerate(solvers)}
    # Small deterministic solver-specific offsets to separate exact overlaps.
    angle = np.linspace(0, 2 * np.pi, len(solvers), endpoint=False)
    solver_unit_offset = {
        s: (float(np.cos(angle[i])), float(np.sin(angle[i])))
        for s, i in solver_index.items()
    }
    dataset_markers = {"mnist": "o", "cifar10": "s"}
    dataset_offsets = {
        "mnist": (-0.01, -0.01),
        "cifar10": (0.01, 0.01),
    }
 
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10.5), dpi=170)
 
    # Panel 1: Distortion targeted vs untargeted
    ax = axes[0]
    dist_pairs = [
        (r["t_l2"], r["u_l2"]) for r in rows if valid(r["t_l2"]) and valid(r["u_l2"])
    ]
    if dist_pairs:
        vals = np.array(dist_pairs)
        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        pad = 0.05 * (mx - mn if mx > mn else 1.0)
        lo, hi = mn - pad, mx + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        x_span = hi - lo
        y_span = hi - lo
    else:
        x_span = 1.0
        y_span = 1.0
    solver_jitter_x = 0.012 * x_span
    solver_jitter_y = 0.012 * y_span
 
    for r in rows:
        if not (valid(r["t_l2"]) and valid(r["u_l2"])):
            continue
        dox = dataset_offsets[r["dataset"]][0] * x_span
        doy = dataset_offsets[r["dataset"]][1] * y_span
        sox = solver_unit_offset[r["solver"]][0] * solver_jitter_x
        soy = solver_unit_offset[r["solver"]][1] * solver_jitter_y
        ox = dox + sox
        oy = doy + soy
        ax.scatter(
            r["t_l2"] + ox,
            r["u_l2"] + oy,
            color=solver_colors[r["solver"]],
            marker=dataset_markers[r["dataset"]],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
            s=85,
        )
 
    ax.set_title("Total L2 Distortion\n(untargeted vs targeted)")
    ax.set_xlabel("Targeted L2 Distortion")
    ax.set_ylabel("Untargeted L2 Distortion")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.text(
        0.03,
        0.97,
        "Below diagonal = untargeted lower",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
 
    # Panel 2: Queries targeted vs untargeted
    ax = axes[1]
    query_pairs = [
        (r["t_q"], r["u_q"]) for r in rows if valid(r["t_q"]) and valid(r["u_q"])
    ]
    if query_pairs:
        vals = np.array(query_pairs)
        mn = float(np.nanmin(vals))
        mx = float(np.nanmax(vals))
        pad = 0.05 * (mx - mn if mx > mn else 1.0)
        lo, hi = mn - pad, mx + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        x_span = hi - lo
        y_span = hi - lo
    else:
        x_span = 1.0
        y_span = 1.0
    solver_jitter_x = 0.012 * x_span
    solver_jitter_y = 0.012 * y_span
 
    for r in rows:
        if not (valid(r["t_q"]) and valid(r["u_q"])):
            continue
        dox = dataset_offsets[r["dataset"]][0] * x_span
        doy = dataset_offsets[r["dataset"]][1] * y_span
        sox = solver_unit_offset[r["solver"]][0] * solver_jitter_x
        soy = solver_unit_offset[r["solver"]][1] * solver_jitter_y
        ox = dox + sox
        oy = doy + soy
        ax.scatter(
            r["t_q"] + ox,
            r["u_q"] + oy,
            color=solver_colors[r["solver"]],
            marker=dataset_markers[r["dataset"]],
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
            s=85,
        )
 
    ax.set_title("Query Efficiency\n(untargeted vs targeted)")
    ax.set_xlabel("Targeted mean queries on success")
    ax.set_ylabel("Untargeted mean queries on success")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.text(
        0.03,
        0.97,
        "Below diagonal = untargeted fewer queries",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
 
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=solver_colors[s],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=7,
            label=s,
        )
        for s in solvers
    ]
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        title="Optimizer (color)",
        fontsize=10,
        title_fontsize=11,
    )
 
    dataset_handles = [
        Line2D(
            [0], [0],
            marker=dataset_markers[d],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="None",
            markersize=7,
            label=d.upper(),
        )
        for d in DATASETS
    ]
    fig.legend(
        handles=dataset_handles,
        labels=[h.get_label() for h in dataset_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        title="Dataset (marker)",
        fontsize=10,
        title_fontsize=11,
    )
 
    plt.tight_layout(rect=[0, 0.12, 1, 1])
 
    out_path = os.path.join("plots", args.output)
    plt.savefig(out_path, bbox_inches="tight")
    print(out_path)
 
 
if __name__ == "__main__":
    main()
 
