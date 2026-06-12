# NES Black-Box Adversarial Attack

Implementation of the NES (Natural Evolution Strategies) black-box adversarial attack from [Ilyas et al., 2018](https://arxiv.org/abs/1804.08598), supporting MNIST, CIFAR-10, and ImageNet with multiple optimizers.

---

## Installation

```bash
pip install -r requirements.txt
```

Then download datasets and pre-trained models:

```bash
python setup.py
```

---

## Running attacks

### Reproducing the paper results

The results reported in the paper come from sweeping every solver across every model, both untargeted and targeted. A single call runs the whole sweep:

```bash
python run.py                  # all solvers, all models, BOTH untargeted + targeted
```

`run.py` calls the attack below once per (dataset, solver, mode) combination and writes each run's output to its own folder (see [Results](#results)). It also renders the targeted-vs-untargeted summary graph (see [Plots](#plots)) at the end; add `--plot-only` to regenerate just the graph without re-running the attacks. Restrict the sweep with `--modes untargeted` or `--modes targeted`, narrow it with `--datasets` / `--solvers` / `--samples`, or pass `--dry-run` to preview the commands first.


### Basic usage

```bash
python nes_attack.py --dataset <mnist|cifar10|imagenet> --solver <solver> --samples <n>
```

### All solvers — MNIST

```bash
python nes_attack.py --dataset mnist --solver momentum --samples 10
python nes_attack.py --dataset mnist --solver nesterov --samples 10
python nes_attack.py --dataset mnist --solver adagrad  --samples 10
python nes_attack.py --dataset mnist --solver adam     --samples 10
python nes_attack.py --dataset mnist --solver sgd      --samples 10
python nes_attack.py --dataset mnist --solver sgdsign  --samples 10
python nes_attack.py --dataset mnist --solver signum   --samples 10
python nes_attack.py --dataset mnist --solver lion     --samples 10
python nes_attack.py --dataset mnist --solver newton   --samples 10
python nes_attack.py --dataset mnist --solver adahessian --samples 10
```

### All solvers — CIFAR-10

```bash
python nes_attack.py --dataset cifar10 --solver momentum --samples 10
python nes_attack.py --dataset cifar10 --solver nesterov --samples 10
python nes_attack.py --dataset cifar10 --solver adagrad  --samples 10
python nes_attack.py --dataset cifar10 --solver adam     --samples 10
python nes_attack.py --dataset cifar10 --solver sgd      --samples 10
python nes_attack.py --dataset cifar10 --solver sgdsign  --samples 10
python nes_attack.py --dataset cifar10 --solver signum   --samples 10
python nes_attack.py --dataset cifar10 --solver lion     --samples 10
python nes_attack.py --dataset cifar10 --solver newton   --samples 10
python nes_attack.py --dataset cifar10 --solver adahessian --samples 10
```

### Targeted attack

```bash
python nes_attack.py --dataset mnist   --solver lion --samples 10 --targeted
python nes_attack.py --dataset cifar10 --solver lion --samples 10 --targeted
```

---

## Early stopping

The attack always stops as soon as the adversarial example successfully fools the model. This measures how many queries each optimizer needs to break the model, which is a strong indicator of solver efficiency. The number of queries used is saved in `results.json`.

---

## All arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | — | `mnist`, `cifar10`, or `imagenet` |
| `--solver` | `momentum` | `momentum`, `nesterov`, `adagrad`, `adam`, `sgd`, `sgdsign`, `signum`, `lion`, `newton`, `adahessian` |
| `--samples` | `10` | Number of images to attack |
| `--targeted` | `False` | Targeted attack (default: untargeted) |
| `--epsilon` | auto | L-inf perturbation budget |
| `--sigma` | auto | NES noise standard deviation |
| `--n_samples` | auto | Number of NES perturbation samples per step |
| `--max_iter` | auto | Maximum number of iterations |
| `--max_lr` | auto | Initial learning rate |
| `--min_lr` | auto | Minimum learning rate |
| `--plateau_length` | `5` | Steps before LR decay |
| `--plateau_drop` | `2.0` | LR decay factor |

Dataset-specific defaults:

| Dataset | epsilon | max_iter | max_lr |
|---|---|---|---|
| MNIST | 0.3 | 500 | 0.05 |
| CIFAR-10 | 0.05 | 500 | 0.01 |
| ImageNet | 0.05 | 200 | 0.01 |

---

## Results

Results are saved in `nes_results/<dataset>/<targeted|untargeted>/<solver>/`:

```
nes_results/
  mnist/untargeted/lion/
    results.json               : success rate, queries, distortion, PSNR, SSIM, time
    original_0.png             : original image
    adversarial_0.png          : adversarial image
    grid_mnist_untargeted_lion.png  : side-by-side grid with class labels
```

`results.json` includes a `queries` block:

```json
"queries": {
  "per_sample": [1200, 800, ...],
  "mean_on_success": 1050.0,
  "budget": 50000
}
```
---
## Plots

The targeted-vs-untargeted summary (report Fig. 4) is produced by `plot_targeted_vs_untargeted_summary.py`, which plots two diagonal comparisons across optimizers: total L2 distortion (untargeted vs targeted) and mean queries on success (untargeted vs targeted). It reads `total_distortion` and `queries.mean_on_success` from each `nes_results/<dataset>/<targeted|untargeted>/<solver>/results.json` (override the root with `--results-dir`), and writes the figure to `plots/`.

`run.py` renders this graph automatically after the sweep finishes. To regenerate it without re-running the attacks (e.g. to test plot generation), use `--plot-only`:

```bash
python run.py --plot-only
```

You can also call the plot script directly, e.g. to drop solvers from the figure:

```bash
python plot_targeted_vs_untargeted_summary.py --exclude-solvers newton adahessian
```

The summary compares MNIST and CIFAR-10 (matching the report figure); it needs both the untargeted and targeted runs present for each optimizer, which a default `run.py` sweep produces.

---

## Sample results

Every run writes a labelled grid image (`grid_<dataset>_<mode>_<solver>.png`) into `nes_results/<dataset>/<targeted|untargeted>/<solver>/`, showing each source image with its `original → adversarial` prediction — open one to see the results visually. To showcase examples directly in this README, drop a few of those grids into a `sample_results/` folder and reference them here, e.g. for CIFAR10, Adam, targeted:

![CIFAR-10 adversarial examples — Adam, targeted](sample_results/grid_cifar10_targeted_adam.png)

