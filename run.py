#!/usr/bin/env python3
"""
run.py — run the ZOO black-box attack with every solver on every model.

Sweeps all coordinate-descent solvers across the chosen datasets by invoking
zoo_l2_attack_black.py once per (dataset, solver) combination.

Examples
--------
    python run.py                          # all solvers, all models, BOTH untargeted + targeted
    python run.py --samples 25             # 25 samples each
    python run.py --modes untargeted       # only the untargeted sweep
    python run.py --modes targeted         # only the targeted sweep
    python run.py --early-stop             # stop each sample once the attack succeeds
    python run.py --datasets cifar10       # only cifar10
    python run.py --solvers adam newton    # only a subset of solvers
    python run.py --datasets imagenet --imagenet_dir /path/to/val
    python run.py --plot-only              # skip the sweep, just (re)generate the charts
    python run.py --datasets imagenet --imagenet_dir /path/to/val
"""
 
import argparse
import subprocess
import sys
import time
 
ATTACK_SCRIPT = "zoo_l2_attack_black.py"
 
# Must match the --solver choices in zoo_l2_attack_black.py
ALL_SOLVERS = ["adam", "newton", "sgd", "sgdsign", "signum", "lion", "adahessian"]
 
# Per-metric bar chart (report Fig. 5), rendered automatically after the sweep.
PLOT_SCRIPT = "plot_zoo_metrics_summary.py"
 
# All models are run by default. imagenet downloads ImageNette (~100 MB) on
# first use and relies on setup_imagenet_model.py being present on the branch.
ALL_DATASETS = ["mnist", "cifar10", "imagenet"]
DEFAULT_DATASETS = ALL_DATASETS
 
 
def render_plots(datasets, modes, results_dir="sample_results"):
    print("\n" + "=" * 70)
    print("PLOTTING per-metric bar charts via %s" % PLOT_SCRIPT)
    print("=" * 70)
    for dataset in datasets:
        for mode in modes:
            subprocess.run([sys.executable, PLOT_SCRIPT,
                            "--dataset", dataset, "--attack-type", mode,
                            "--results-dir", results_dir])
 
 
def main():
    p = argparse.ArgumentParser(description="Run the ZOO attack with all solvers on all models.")
    p.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=DEFAULT_DATASETS,
                   help="Datasets to attack (default: mnist cifar10 imagenet)")
    p.add_argument("--solvers", nargs="+", choices=ALL_SOLVERS, default=ALL_SOLVERS,
                   help="Solvers to run (default: all)")
    p.add_argument("--samples", type=int, default=10,
                   help="Number of SOURCE images per run (default: 10). Untargeted: N attacks. "
                        "Targeted: each source is attacked toward every other class, so for "
                        "10-class MNIST/CIFAR-10 that is N x 9 (e.g. 10 -> 90 attacks).")
    p.add_argument("--modes", nargs="+", choices=["untargeted", "targeted"],
                   default=["untargeted", "targeted"],
                   help="Attack modes to run (default: both untargeted and targeted)")
    p.add_argument("--early-stop", action="store_true",
                   help="Stop each sample once the attack succeeds")
    p.add_argument("--imagenet_dir", default=None,
                   help="Path to ImageNet val directory (used when imagenet is in --datasets)")
    p.add_argument("--results-dir", default="sample_results",
                   help="Root directory holding the <dataset>/<mode>/<solver>/ results tree, "
                        "and where the charts read from (default: sample_results)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip the attacks and only (re)generate the charts from existing "
                        "results.json files, handy for testing plot generation")
    args = p.parse_args()
 
    if args.plot_only:
        render_plots(args.datasets, args.modes, args.results_dir)
        return
 
    runs = [(d, s, m) for d in args.datasets for s in args.solvers for m in args.modes]
    print("Planned runs: %d  (%d datasets x %d solvers x %d modes)\n"
          % (len(runs), len(args.datasets), len(args.solvers), len(args.modes)))
 
    results = []
    t0 = time.time()
    for i, (dataset, solver, mode) in enumerate(runs, 1):
        cmd = [sys.executable, ATTACK_SCRIPT,
               "--dataset", dataset, "--solver", solver,
               "--samples", str(args.samples)]
        if mode == "targeted":
            cmd.append("--targeted")
        if args.early_stop:
            cmd.append("--early-stop")
        if dataset == "imagenet" and args.imagenet_dir:
            cmd += ["--imagenet_dir", args.imagenet_dir]
 
        print("=" * 70)
        print("[%d/%d] %s" % (i, len(runs), " ".join(cmd)))
        print("=" * 70)
 
        if args.dry_run:
            continue
 
        rc = subprocess.run(cmd).returncode
        results.append((dataset, solver, mode, rc))
        if rc != 0:
            print("  ! run failed (exit %d) — continuing" % rc, file=sys.stderr)
 
    if args.dry_run:
        return
 
    print("\n" + "=" * 70)
    print("SUMMARY  (%.1f mins total)" % ((time.time() - t0) / 60.0))
    print("=" * 70)
    ok = sum(1 for *_, rc in results if rc == 0)
    for dataset, solver, mode, rc in results:
        print("  %-9s %-11s %-11s %s" % (dataset, solver, mode, "ok" if rc == 0 else "FAILED (%d)" % rc))
    print("\n%d/%d runs succeeded" % (ok, len(results)))
 
    render_plots(args.datasets, args.modes, args.results_dir)
 
    sys.exit(0 if ok == len(results) else 1)
 
 
if __name__ == "__main__":
    main()
 
