"""
flip_map.py — Per-pixel perturbation analysis between original and adversarial images.

Produces for each sample:
  • diff_raw_{i}.png      — signed L2 per-pixel magnitude (heat-map, RdBu colourmap)
  • diff_overlay_{i}.png  — original image with top-K perturbed pixels highlighted
  • flip_map_summary.png  — mean absolute perturbation aggregated across all samples

Usage:
  python flip_map.py --dir cifar10/untargeted/newton
  python flip_map.py --dir cifar10/untargeted/newton --topk 50 --colormap hot
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

plt.rcParams["font.family"] = "Times New Roman"


# ─────────────────────────────── helpers ────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load PNG as float32 array in [0, 1], shape (H, W, C)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def perturbation_magnitude(orig: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """L2 magnitude of per-pixel perturbation, shape (H, W)."""
    return np.sqrt(np.sum((adv - orig) ** 2, axis=-1))


def signed_diff(orig: np.ndarray, adv: np.ndarray) -> np.ndarray:
    """Mean signed channel difference per pixel, shape (H, W)."""
    return np.mean(adv - orig, axis=-1)


def top_k_mask(mag: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k pixels with largest perturbation magnitude."""
    flat = mag.flatten()
    threshold = np.sort(flat)[-k] if k < len(flat) else flat.min()
    return mag >= threshold


# ─────────────────────────────── per-sample plot ────────────────────────────

def plot_sample(orig: np.ndarray, adv: np.ndarray, idx: int,
                out_dir: str, topk: int, colormap: str,
                orig_label: str = "", adv_label: str = "") -> np.ndarray:
    """
    Saves diff_raw and diff_overlay for one sample.
    Returns the per-pixel magnitude array for global aggregation.
    """
    mag  = perturbation_magnitude(orig, adv)   # (H, W)
    sdiff = signed_diff(orig, adv)             # (H, W)

    # ── 1. raw heat-map ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle(f"Sample {idx}  |  {orig_label} → {adv_label}", fontsize=11)

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(adv)
    axes[1].set_title("Adversarial", fontsize=9)
    axes[1].axis("off")

    vmax = max(mag.max(), 1e-6)
    im = axes[2].imshow(sdiff, cmap=colormap, vmin=-vmax, vmax=vmax)
    axes[2].set_title("Signed Δ (mean channel)", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    raw_path = os.path.join(out_dir, f"diff_raw_{idx}.png")
    fig.savefig(raw_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. overlay: original + top-K highlighted pixels ─────────────────────
    mask = top_k_mask(mag, topk)

    overlay = orig.copy()
    # Highlight: set masked pixels to pure red
    overlay[mask] = [1.0, 0.0, 0.0]

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(overlay)
    red_patch = mpatches.Patch(color="red", label=f"Top-{topk} perturbed pixels")
    ax2.legend(handles=[red_patch], fontsize=7, loc="lower right")
    ax2.set_title(f"Sample {idx}: {orig_label} → {adv_label}", fontsize=9)
    ax2.axis("off")
    plt.tight_layout()
    ov_path = os.path.join(out_dir, f"diff_overlay_{idx}.png")
    fig2.savefig(ov_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return mag   # returned for global aggregation


# ─────────────────────────────── summary plot ───────────────────────────────

def plot_summary(mean_mag: np.ndarray, out_dir: str, colormap: str,
                 solver: str, dataset: str, targeted: bool) -> None:
    """Saves a single summary heat-map of mean perturbation magnitude."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mean_mag, cmap=colormap)
    targeted_str = "targeted" if targeted else "untargeted"
    ax.set_title(f"Mean |Δ| — {dataset} / {targeted_str} / {solver}", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="L2 magnitude")
    plt.tight_layout()
    path = os.path.join(out_dir, "flip_map_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary saved → {path}")


# ─────────────────────────────── channel-wise ───────────────────────────────

def plot_channel_maps(orig: np.ndarray, adv: np.ndarray, idx: int,
                      out_dir: str, colormap: str) -> None:
    """Per-channel signed difference maps (R, G, B)."""
    channel_names = ["R", "G", "B"]
    n = orig.shape[2]
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]
    diff = adv - orig   # (H, W, C)
    vmax = max(np.abs(diff).max(), 1e-6)
    for c, ax in enumerate(axes):
        im = ax.imshow(diff[:, :, c], cmap=colormap, vmin=-vmax, vmax=vmax)
        ax.set_title(f"Channel {channel_names[c] if c < 3 else c}", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Sample {idx} — per-channel Δ", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f"diff_channels_{idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────── main ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Flip-map analysis of adversarial perturbations")
    parser.add_argument("--dir",      required=True,
                        help="Path to attack output directory, e.g. cifar10/untargeted/newton")
    parser.add_argument("--topk",     type=int, default=30,
                        help="Number of top-perturbed pixels to highlight in overlay (default: 30)")
    parser.add_argument("--colormap", default="RdBu_r",
                        help="Matplotlib colourmap for difference plots (default: RdBu_r)")
    parser.add_argument("--channels", action="store_true",
                        help="Also save per-channel R/G/B difference maps")
    args = parser.parse_args()

    attack_dir = args.dir
    if not os.path.isdir(attack_dir):
        raise FileNotFoundError(f"Directory not found: {attack_dir}")

    # ── Load results.json for labels ─────────────────────────────────────────
    results_path = os.path.join(attack_dir, "results.json")
    valid_labels = adv_labels = None
    dataset = solver = "unknown"
    targeted = False
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        valid_labels = results.get("valid_classification", None)
        adv_labels   = results.get("adversarial_classification", None)
        dataset      = results.get("dataset", "unknown")
        solver       = results.get("solver", "unknown")
        targeted     = results.get("targeted", False)

    # CIFAR-10 class names (fallback to index strings for MNIST)
    cifar_classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    def label_name(idx_val, dataset_name):
        if dataset_name == "cifar10" and idx_val is not None:
            return cifar_classes[int(idx_val)]
        return str(idx_val) if idx_val is not None else "?"

    # ── Discover sample pairs ────────────────────────────────────────────────
    orig_files = sorted(
        [f for f in os.listdir(attack_dir) if f.startswith("original_") and f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    if not orig_files:
        raise FileNotFoundError(f"No original_*.png files found in {attack_dir}")

    out_dir = os.path.join(attack_dir, "flip_maps")
    os.makedirs(out_dir, exist_ok=True)

    all_mags = []

    for fname in orig_files:
        i = int(fname.split("_")[1].split(".")[0])
        adv_fname = f"adversarial_{i}.png"
        orig_path = os.path.join(attack_dir, fname)
        adv_path  = os.path.join(attack_dir, adv_fname)

        if not os.path.exists(adv_path):
            print(f"  [skip] no adversarial file for sample {i}")
            continue

        orig = load_image(orig_path)
        adv  = load_image(adv_path)

        orig_lbl = label_name(valid_labels[i] if valid_labels else None, dataset)
        adv_lbl  = label_name(adv_labels[i]   if adv_labels   else None, dataset)

        print(f"  Sample {i}: {orig_lbl} → {adv_lbl}")

        mag = plot_sample(orig, adv, i, out_dir,
                          topk=args.topk, colormap=args.colormap,
                          orig_label=orig_lbl, adv_label=adv_lbl)
        all_mags.append(mag)

        if args.channels:
            plot_channel_maps(orig, adv, i, out_dir, args.colormap)

    if not all_mags:
        print("No samples processed.")
        return

    # ── Summary across all samples ───────────────────────────────────────────
    mean_mag = np.mean(np.stack(all_mags, axis=0), axis=0)   # (H, W)
    plot_summary(mean_mag, out_dir, args.colormap, solver, dataset, targeted)

    # ── Print top-10 most perturbed pixel coordinates ────────────────────────
    flat_idx = np.argsort(mean_mag.flatten())[::-1][:10]
    H, W = mean_mag.shape
    print("\nTop-10 most perturbed pixels (row, col) — mean magnitude:")
    for rank, fi in enumerate(flat_idx):
        r, c = divmod(fi, W)
        print(f"  #{rank+1:2d}  pixel ({r:3d}, {c:3d})  |Δ| = {mean_mag[r, c]:.5f}")

    print(f"\nAll flip maps saved in: {out_dir}")


if __name__ == "__main__":
    main()
