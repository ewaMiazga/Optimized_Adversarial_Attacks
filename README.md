# PGD White-Box Adversarial Attack

Implementation of the PGD (Projected Gradient Descent) white-box adversarial attack from [Madry et al., 2018](https://arxiv.org/abs/1706.06083), supporting MNIST, CIFAR-10, and ImageNet with multiple optimizers.

The default `sgdsign` solver is classic PGD. The other solvers let each optimizer consume the exact gradient here, so it can be compared against the same optimizer running on the estimated gradient in the ZOO and NES black-box attacks.

---

## Installation

```bash
pip install -r pgd_venv.txt
```

The pre-trained MNIST and CIFAR-10 models are already included in `models/`. The MNIST and CIFAR-10 datasets are downloaded automatically by torchvision on first run, and the ImageNette subset used for ImageNet is fetched automatically when `--dataset imagenet` is selected.

---

## Running attacks

### Reproducing the paper results

The results reported in the paper come from sweeping every solver across every model, both untargeted and targeted. A single call runs the whole sweep:

```bash
python run.py   # all solvers, all models, both untargeted + targeted
```

`run.py` calls the attack below once per (dataset, solver, mode) combination and writes each run's output to its own folder. Restrict the sweep with `--modes untargeted` or `--modes targeted`, narrow it with `--datasets` / `--solvers` / `--samples`, or pass `--dry-run` to preview the commands first.

### Basic usage

```bash
python pgd_attack.py --dataset <mnist|cifar10|imagenet> --solver <solver> --samples <n>
```

### All solvers — MNIST

```bash
python pgd_attack.py --dataset mnist --solver sgdsign    --samples 10
python pgd_attack.py --dataset mnist --solver sgd        --samples 10
python pgd_attack.py --dataset mnist --solver adam       --samples 10
python pgd_attack.py --dataset mnist --solver signum     --samples 10
python pgd_attack.py --dataset mnist --solver lion       --samples 10
python pgd_attack.py --dataset mnist --solver newton     --samples 10
python pgd_attack.py --dataset mnist --solver adahessian --samples 10
```

### All solvers — CIFAR-10

```bash
python pgd_attack.py --dataset cifar10 --solver sgdsign    --samples 10
python pgd_attack.py --dataset cifar10 --solver sgd        --samples 10
python pgd_attack.py --dataset cifar10 --solver adam       --samples 10
python pgd_attack.py --dataset cifar10 --solver signum     --samples 10
python pgd_attack.py --dataset cifar10 --solver lion       --samples 10
python pgd_attack.py --dataset cifar10 --solver newton     --samples 10
python pgd_attack.py --dataset cifar10 --solver adahessian --samples 10
```

### Targeted attack

```bash
python pgd_attack.py --dataset mnist   --solver sgdsign --samples 10 --targeted
python pgd_attack.py --dataset cifar10 --solver sgdsign --samples 10 --targeted
```

---

## Early stopping

The attack always stops as soon as the adversarial example successfully fools the model. This measures how many queries each optimizer needs to break the model, which is a strong indicator of solver efficiency. The number of queries used is saved in `results.json`, where one PGD iteration counts as 3 queries (one forward loss, one backward gradient, one forward success-check).

---

## All arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | `mnist`, `cifar10`, or `imagenet` |
| `--solver` | `sgdsign` | `sgd`, `sgdsign`, `adam`, `signum`, `lion`, `newton`, `adahessian` (`sgdsign` is classic PGD) |
| `--samples` | `10` | Number of **source** images. Untargeted: N attacks. Targeted: each source is attacked toward every other class, so for 10-class MNIST/CIFAR-10 that's N × 9 (e.g. 10 → 90 attacks) |
| `--start` | `6` | Offset into the test set (same as ZOO / NES) |
| `--targeted` | `False` | Targeted attack (default: untargeted) |
| `--targeted-k` | `None` | Number of non-true target classes per source image in targeted mode |
| `--target-label-set` | `all` | Target class pool for targeted attacks: `all` or `imagenette10` |
| `--targeted-classes` | `None` | Comma-separated class IDs to use as targets (overrides `--target-label-set`) |
| `--imagenette-one-per-class` | `False` | For ImageNet: one correctly-classified sample per ImageNette class (10 sources) |
| `--imagenet_dir` | `./mini_imagenet` | Path to ImageNet val directory |
| `--epsilon` | auto | L-inf perturbation budget |
| `--step_size` | auto | PGD sign-step size (alpha) |
| `--max_iter` | auto | Maximum number of iterations |
| `--seed` | `42` | Random seed for reproducible sample ordering |

Dataset-specific defaults:

| Dataset | epsilon | step_size | max_iter |
|---|---|---|---|
| MNIST | 0.3 | 0.075 | 500 |
| CIFAR-10 | 0.05 | 0.0125 | 500 |
| ImageNet | 0.05 | 0.0125 | 200 |

---

## Results

Results are saved in `<dataset>/<targeted|untargeted>/pgd_<solver>/`:

```
cifar10/
  untargeted/pgd_sgdsign/
    results.json               ← success rate, queries, distortion, PSNR, SSIM, time
    original_0.png             ← original image
    adversarial_0.png          ← adversarial image
    grid_cifar10_untargeted_pgd_sgdsign.png  ← side-by-side grid with class labels
```

`results.json` includes a `queries` block:

```json
"queries": {
  "per_sample": [1200, 800, ...],
  "mean_on_success": 1050.0,
  "counting_convention": "3 per PGD iter (1 fwd loss + 1 bwd grad + 1 fwd success-check)"
}
```

along with `success_rate_pct`, `total_distortion`, `time_mins`, a `distortion` block (per-sample and mean L-inf / L2 on success), the per-sample `mse`, `mae`, `psnr`, and `ssim` blocks, and an `attack_params` block recording `epsilon`, `step_size`, and `max_iter`.
