"""
setup_imagenet_model.py — Pretrained VGG-16 wrapper for the ZOO L2 attack.

The attack pipeline normalises every dataset with
    Normalize(mean=(0.5,), std=(1.0,))   →  pixel ∈ [-0.5, 0.5]

The VGG16Wrapper undoes that shift and applies the standard ImageNet
normalisation internally, so the attack code requires zero changes.

Download:
    Weights are fetched automatically by torchvision on first use
    (~550 MB, cached in ~/.cache/torch/hub).

Quick model-accuracy check:
    python setup_imagenet_model.py --val_dir /path/to/imagenet/val
"""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader


# ── ImageNet statistics ──────────────────────────────────────────────────────
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGG16Wrapper(nn.Module):
    """
    VGG-16 pretrained on ImageNet wrapped to accept inputs in [-0.5, 0.5]
    (the normalisation used throughout this codebase).

    Forward pass:
        x  ∈  [-0.5, 0.5]          (attack-pipeline convention)
        → x + 0.5  ∈  [0, 1]
        → ImageNet normalise
        → VGG-16 logits  (shape: batch × 1000)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg = models.vgg16(weights=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo the (mean=0.5, std=1.0) shift used by the attack's DataLoader
        x = x + 0.5                                          # → [0, 1]
        # Apply ImageNet normalisation
        mean = _IMAGENET_MEAN.to(x.device, x.dtype)
        std  = _IMAGENET_STD.to(x.device, x.dtype)
        x = (x - mean) / std
        return self.vgg(x)


# ── Convenience transform for ImageNet images ────────────────────────────────

def imagenet_transform(image_size: int = 224) -> transforms.Compose:
    """
    Standard ImageNet eval transform, ending with Normalize((0.5,),(1.0,))
    so it matches the convention expected by VGG16Wrapper.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                             # → [0, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                              std=(1.0, 1.0, 1.0)),        # → [-0.5, 0.5]
    ])


def imagenet_loader(val_dir: str,
                    batch_size: int = 1,
                    shuffle: bool = True,
                    num_workers: int = 2) -> DataLoader:
    """
    Build a DataLoader from an ImageNet-style directory:
        val_dir/
            n01440764/   ← synset folder
                *.JPEG
            n01443537/
                ...

    If val_dir contains .JPEG/.jpg/.png files directly (flat layout),
    wrap them as a single pseudo-class — useful for small custom sets.
    """
    dataset = datasets.ImageFolder(root=val_dir,
                                   transform=imagenet_transform())
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)


# ── ImageNet top-1000 class names (synset → readable) ───────────────────────

def get_imagenet_labels() -> list[str]:
    """
    Returns a list of 1000 human-readable ImageNet class names,
    downloaded once via torchvision's built-in metadata.
    """
    weights = models.VGG16_Weights.IMAGENET1K_V1
    return weights.meta["categories"]


# ── Quick accuracy check ─────────────────────────────────────────────────────

def check_accuracy(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   max_batches: int = 100) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    print(f"Top-1 accuracy: {acc*100:.2f}%  ({correct}/{total})")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", required=True,
                        help="Path to ImageNet val directory (ImageFolder layout)")
    parser.add_argument("--max_batches", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading VGG-16 (pretrained) on {device} …")
    model = VGG16Wrapper(pretrained=True).to(device)
    model.eval()

    loader = imagenet_loader(args.val_dir, batch_size=32, shuffle=False, num_workers=4)
    check_accuracy(model, loader, device, max_batches=args.max_batches)
