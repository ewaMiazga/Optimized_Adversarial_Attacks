#!/usr/bin/env python3
"""
Run the NES black-box attack with every solver on every model.

Sweeps all optimizers across the chosen datasets by invoking nes_attack.py
once per (dataset, solver) combination, then renders the targeted-vs-untargeted
summary graph (report Fig. 4) from the produced results.

Examples
--------
    python run.py                          # all solvers, all models, both untargeted + targeted, then plot
    python run.py --samples 25             # 25 samples each
    python run.py --modes untargeted       # only the untargeted sweep
    python run.py --modes targeted         # only the targeted sweep
    python run.py --datasets cifar10       # only cifar10
    python run.py --solvers adam lion      # only a subset of solvers
    python run.py --plot-only              # skip the sweep, just (re)generate the graph
    python run.py --imagenet_dir /path/to/val
"""

import argparse
import subprocess
import sys
import time

ATTACK_SCRIPT = "nes_attack.py"

# Targeted-vs-untargeted summary plot (report Fig. 4), rendered after the sweep.
# Reads nes_results/<dataset>/<targeted|untargeted>/<solver>/results.json
PLOT_SCRIPT = "plot_targeted_vs_untargeted_summary.py"

# Must match the --solver choices in nes_attack.py
ALL_SOLVERS = ["momentum", "nesterov", "adagrad", "adam", "sgd",
               "sgdsign", "signum", "lion", "newton", "adahessian"]

# All models are run by default. imagenet downloads ImageNette (~100 MB) on
# first use and relies on setup_imagenet_model.py being present on the branch.
ALL_DATASETS = ["mnist", "cifar10", "imagenet"]
DEFAULT_DATASETS = ALL_DATASETS


def render_plot():
    print("\n" + "=" * 70)
    print("PLOTTING targeted-vs-untargeted summary via %s" % PLOT_SCRIPT)
    print("=" * 70)
    subprocess.run([sys.executable, PLOT_SCRIPT])


def main():
    p = argparse.ArgumentParser(description="Run the NES attack with all solvers on all models.")
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
    p.add_argument("--imagenet_dir", default=None,
                   help="Path to ImageNet val directory (used when imagenet is in --datasets)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without running them")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip the attacks and only (re)generate the summary graph from existing "
                        "results.json files — handy for testing plot generation")
    args = p.parse_args()

    if args.plot_only:
        render_plot()
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
            print("  ! run failed (exit %d) - continuing" % rc, file=sys.stderr)

    if args.dry_run:
        return

    print("\n" + "=" * 70)
    print("SUMMARY  (%.1f mins total)" % ((time.time() - t0) / 60.0))
    print("=" * 70)
    ok = sum(1 for *_, rc in results if rc == 0)
    for dataset, solver, mode, rc in results:
        print("  %-9s %-11s %-11s %s" % (dataset, solver, mode, "ok" if rc == 0 else "FAILED (%d)" % rc))
    print("\n%d/%d runs succeeded" % (ok, len(results)))

    render_plot()

    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()