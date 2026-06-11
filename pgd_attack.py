"""
PGD white-box attack, from Madry et al., 2018: "Towards Deep Learning Models Resistant to Adversarial Attacks" 
https://doi.org/10.48550/arXiv.1706.06083

run via the following commands:
python pgd_attack.py --dataset cifar10 --samples 10
  python pgd_attack.py --dataset mnist   --samples 10
  python pgd_attack.py --dataset cifar10 --samples 10 --targeted

more flags exist, check main!

output directory: 
<dataset>/<targeted|untargeted>/pgd/
"""

import json
import time
import argparse
import os
import sys
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim

from setup_mnist_model    import MNIST
from setup_cifar10_model  import CIFAR10

# ImageNette synset folders mapped to true ImageNet class IDs
IMAGENETTE_TO_IMAGENET = {
    'n01440764': 0,    # tench
    'n02102040': 217,  # English springer
    'n02979186': 482,  # cassette player
    'n03000684': 491,  # chain saw
    'n03028079': 497,  # church
    'n03394916': 566,  # French horn
    'n03417042': 569,  # garbage truck
    'n03425413': 571,  # gas pump
    'n03445777': 574,  # golf ball
    'n03888257': 701,  # parachute
}
IMAGENETTE_LABEL_IDS = list(IMAGENETTE_TO_IMAGENET.values())

## copying zoo l2 attack dataset handling, images normalized to [-0.5, 0.5]
def load_dataset_and_model(dataset_name, device, imagenet_dir=None, loader_seed=42):

    # from zoo_l2_attack_black: Normalize((0.5,),(1.0,)) for every dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,)),   # [-0.5, 0.5]
    ])

    if dataset_name == 'mnist':
        test_set = datasets.MNIST(root='./data', train=False,
                                  transform=transform, download=True)
        model = MNIST().to(device)
        model.load_state_dict(torch.load(
            './models/mnist_model.pt', map_location=device, weights_only=False))
        num_labels  = 10
        label_names = [str(i) for i in range(10)]

    elif dataset_name == 'cifar10':
        test_set = datasets.CIFAR10(root='./data', train=False,
                                    transform=transform, download=True)
        model = CIFAR10().to(device)
        model.load_state_dict(torch.load(
            './models/cifar10_model.pt', map_location=device, weights_only=False))
        num_labels  = 10
        label_names = ['plane','car','bird','cat','deer',
                       'dog','frog','horse','ship','truck']

    elif dataset_name == 'imagenet':
        from torchvision import models
        from torchvision.datasets import ImageFolder

        val_dir = os.path.join('data', 'imagenette2-320', 'val')
        if not os.path.isdir(val_dir):
            import urllib.request, tarfile
            os.makedirs('data', exist_ok=True)
            archive = os.path.join('data', 'imagenette2-320.tgz')
            if not os.path.exists(archive):
                print('ImageNette not found, downloading (~100 MB)...')
                urllib.request.urlretrieve(
                    'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
                    archive,
                    reporthook=lambda b, bs, t: print(
                        '\r  %.1f/%.1f MB' % (b*bs/1e6, t/1e6), end='', flush=True))
                print()
            print('Extracting...')
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall('data')
            os.remove(archive)
            print('ImageNette ready.')

        img_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])

        class _ImageNetteDataset(ImageFolder):
            """ImageFolder with true ImageNet class IDs from synset folder names."""
            def __getitem__(self, idx):
                img, folder_idx = super().__getitem__(idx)
                synset = self.classes[folder_idx]
                label  = IMAGENETTE_TO_IMAGENET.get(synset, 0)
                return img, label

        test_set = _ImageNetteDataset(val_dir, transform=img_transform)

        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        inception.aux_logits = False

        class _InceptionWrapper(torch.nn.Module):
            """Accepts [-0.5,0.5] input, applies ImageNet normalisation inside."""
            _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            def __init__(self, base): super().__init__(); self.base = base
            def forward(self, x):
                x = x + 0.5                                  # → [0, 1]
                mean = self._MEAN.to(x.device, x.dtype)
                std  = self._STD.to(x.device, x.dtype)
                x = (x - mean) / std
                return self.base(x)

        model = _InceptionWrapper(inception).to(device)
        num_labels  = 1000
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        label_names = weights.meta['categories']

    else:
        raise ValueError('Unknown dataset: %s' % dataset_name)

    model.eval()
    #loader_gen = torch.Generator()
    #loader_gen.manual_seed(int(loader_seed))
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)#, generator=loader_gen)
    return loader, model, num_labels, label_names


def choose_device_with_cuda_probe():
    """Pick CUDA only if a minimal GPU forward pass succeeds."""
    if not torch.cuda.is_available():
        return torch.device('cpu'), 'CUDA not available, using CPU'

    probe = (
        "import torch\n"
        "x=torch.randn(1,1,28,28,device='cuda')\n"
        "m=torch.nn.Conv2d(1,8,3).cuda()\n"
        "with torch.no_grad():\n"
        "    _=m(x)\n"
        "torch.cuda.synchronize()\n"
    )

    result = subprocess.run(
        [sys.executable, '-c', probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device('cuda'), 'Using GPU: %s' % gpu_name

    return torch.device('cpu'), (
        'CUDA detected but self-test failed (likely driver/WSL/CUDA runtime mismatch). '
        'Falling back to CPU to avoid Bus error.'
    )


def select_one_per_label(data_arr, label_arr, required_labels, start=0):
  """
  Select one sample per required label, preserving `required_labels` order.
  Returns (selected_data, selected_labels, missing_labels).
  """
  first_idx = {}
  req = set(required_labels)
  start_idx = max(int(start), 0)

  for i in range(start_idx, len(label_arr)):
    lbl = int(label_arr[i])
    if lbl in req and lbl not in first_idx:
      first_idx[lbl] = i
      if len(first_idx) == len(required_labels):
        break

  chosen_labels = [lbl for lbl in required_labels if lbl in first_idx]
  missing = [lbl for lbl in required_labels if lbl not in first_idx]

  if not chosen_labels:
    return np.empty((0, *data_arr.shape[1:]), dtype=data_arr.dtype), np.empty((0,), dtype=label_arr.dtype), missing

  idxs = [first_idx[lbl] for lbl in chosen_labels]
  return data_arr[idxs], label_arr[idxs], missing


## mirror generate_data of zoo l2 attack
def generate_data(loader, targeted, samples, start, num_labels, targeted_k=None,
                  target_class_ids=None):
    """
    target_class_ids : optional list of class IDs to use as targets in targeted mode.
               If None, all num_labels classes (except true) are used.
    targeted_k       : optional number of non-true targets per source image in
               targeted mode. If None, uses all candidates.
    """
    inputs=[]
    targets=[]
    cnt=0 
    for i, (data, label) in enumerate(loader):
        if cnt >= samples:
            break
        if i <= start:
            continue
        x  = data[0].numpy()          
        lbl = int(label.item())
        if targeted:
          class_pool = target_class_ids if target_class_ids is not None else range(num_labels)
          candidate_targets = [j for j in class_pool if j != lbl]

          if not candidate_targets:
            cnt += 1
            continue

          if targeted_k is None:
            selected_targets = candidate_targets
          else:
            k = max(1, min(int(targeted_k), len(candidate_targets)))
            selected_targets = np.random.choice(candidate_targets, size=k, replace=False).tolist()

          for j in selected_targets:
                inputs.append(x)
                targets.append(np.eye(num_labels)[j])
        else:
            inputs.append(x)
            targets.append(np.eye(num_labels)[lbl])
        cnt += 1

    return np.array(inputs), np.array(targets)



def margin_loss(logits, label_idx, targeted, device):
  """C&W-style margin loss,  non-zero until the attack succeeds.
  Untargeted : logit[true]  - max_{j!=true}   logit[j]   (minimise = flip)
  Targeted   : max_{j!=tgt}  logit[j] - logit[tgt]        (minimise = reach)
  logits : (n, num_classes) tensor.  Returns: (n,) per-image loss.
  """
  n = logits.shape[0]
  mask = torch.ones(n, logits.shape[1], dtype=torch.bool, device=device)
  mask[:, label_idx] = False                                  # drop the chosen class
  other_max = logits[mask].view(n, -1).max(dim=1)[0]          # strongest other class
  chosen    = logits[:, label_idx]                            # the true / target class
  if targeted:
    return other_max - chosen          # minimise: target becomes the argmax
  else:
    return chosen - other_max          # minimise: true class stops being argmax


# ── Single-image PGD attack ─────────────────────────────────────────────────
#   PGD :  g = autograd.grad(loss, x): backpropagation
def pgd_attack_one(x_np, target_np, model, targeted, device, epsilon, step_size, max_iter, solver='sgdsign'):
  """Run PGD on one image; stops as soon as the attack succeeds so the query count reflects number of queries
  to break the model, comparable to ZOO/NES early-stop.
  x_np      : (C,H,W) numpy array in normalised space [-0.5, 0.5]
  target_np : (num_classes,) one-hot:true class (untargeted) or target (targeted)
  epsilon   : L-inf perturbation budget   
  step_size : optimizer step size
  solver    : which optimizer consumes the exact gradient (see PGD_SOLVERS)
  """

  x0    = torch.from_numpy(x_np).unsqueeze(0).to(device)     
  adv   = x0.clone()
  lower = (x0 - epsilon).clamp(-0.5, 0.5)                     
  upper = (x0 + epsilon).clamp(-0.5, 0.5)

  label_idx = int(np.argmax(target_np))
  with torch.no_grad():
    orig_idx = int(model(x0).argmax(dim=1).item())           

  loss_hist = []
  queries   = 0
  state      = {} 
  needs_hess = solver in CURVATURE_SOLVERS


  for step in range(max_iter):
    # exact gradient: backprop wrt the image, not the weights!
    
    # Curvature solvers need create_graph=True so the gradient can itself be
    # differentiated again (second derivative is the Hessian-vector product).

    adv_var = adv.clone().detach().requires_grad_(True)       # track grad on the pixels
    logits  = model(adv_var)                              
    loss = margin_loss(logits, label_idx, targeted, device).sum()
    grad,= torch.autograd.grad(loss, adv_var, create_graph = needs_hess)            
    queries += 2 # backward step is one extra query

    hvp_fn = None
    if needs_hess:
      # differentiate (grad dot v) again wrt the imagei via autograd
      # Each call is one extra backward pass, each pass is one query!
      def hvp_fn(v, _grad=grad, _var=adv_var):
        Hv, = torch.autograd.grad(_grad, _var, grad_outputs=v,
                                  retain_graph=True)
        return Hv.detach()
      # hutchinson_diag uses 3 probes! so 3 extra backward passes = + 3 queries
      queries += 3

    loss_hist.append(float(loss.item()))

    # original implemention version of below block:
    # adv = adv - step_size * torch.sign(grad)  # L-inf PGD step
    # adv = adv.clamp(lower, upper).detach() # project onto L-inf ball
    
    # solver_step() returns delta, an uphill-pointing update. it is built from the gradient, which points 
    # toward incresing loss.. We want to minimise the margin loss to fool the model, so we move the image in the
    # -delta direction (aka downhill). solver_step always returns a positive, uphill update, so the minus sign is applied here.
    # Then clamp() projects the image back into the L-inf budgeti as usual for PGD.
    grad_use = grad.detach() if needs_hess else grad
    delta = solver_step(solver, grad_use, state, step_size,hvp_fn=hvp_fn) # chosen optimizer
    adv = adv - delta
    adv = adv.clamp(lower, upper).detach() # project onto L-inf ball
 

    with torch.no_grad():
      pred = int(model(adv).argmax(dim=1).item())
    queries += 1

    success = (pred == label_idx) if targeted else (pred != orig_idx)
    if success:
      print('  Early stop at step %d (%d queries)' % (step + 1, queries))
      adv_np    = adv.squeeze(0).cpu().numpy()
      dist_linf = float(np.max(np.abs(adv_np - x_np)))        # L-inf distortion
      dist_l2   = float(np.sqrt(np.sum((adv_np - x_np) ** 2)))# L2 distortion
      return adv_np, True, dist_linf, dist_l2, loss_hist, queries

    if step % 20 == 0:
      print('  step %04d | loss %.4f' % (step, float(loss.item())))

  # ---- attack did not succeed within max_iter ─────────────────────────────
  with torch.no_grad():
    pred = int(model(adv).argmax(dim=1).item())               # final prediction
  success = (pred == label_idx) if targeted else (pred != orig_idx)

  adv_np    = adv.squeeze(0).cpu().numpy()
  dist_linf = float(np.max(np.abs(adv_np - x_np)))            # L-inf distortion
  dist_l2   = float(np.sqrt(np.sum((adv_np - x_np) ** 2)))    # L2 distortion
  return adv_np, success, dist_linf, dist_l2, loss_hist, queries

# ---- Optimizer implementations ───────────────────────────────────────────────────────────────────────────────────────
"""

"""
PGD_SOLVERS = ['sgd', 'sgdsign', 'adam', 'signum', 'lion', 'newton', 'adahessian']
CURVATURE_SOLVERS = ['newton', 'adahessian']
HESS_FLOOR = 0.1 # from ZOO's coordinate_Newton/AdaHessian functions

def hutchinson_diag(hvp_fn, like, n_probes=3):
  """Estimate the diagonal of the Hessian via Hutchinson's method.
  diag(H) ≈ mean over random ±1 vectors z of  (z dot Hz).
  Hessian-diagonal estimator used by the AdaHessian paper
  (Yao et al., 2021 — arxiv.org/abs/2006.00719)
  """
  diag = torch.zeros_like(like)
  for _ in range(n_probes):
    z  = torch.randint(0, 2, like.shape, device=like.device,
                       dtype=like.dtype) * 2 - 1     # random +/- 1 vector
    Hz = hvp_fn(z)                                    # exact Hessian-vector product
    diag += z * Hz
  return diag / n_probes
 
def solver_step(solver, grad, state, step_size,
                beta1=0.9, beta2=0.999, eps=1e-8, hvp_fn=None):
  """Return the update tensor `delta` for one step. adv ← adv - delta.
  Reproduces the ZOO coordinate_* formulas, fed PGD's exact gradient.
  `state`  : mutable dict carrying ZOO-equivalent buffers (mt, vt, epoch).
  `hvp_fn` : Hessian-vector-product function, required for curvature solvers.
  """
  if solver == 'sgd':
    # SGD: gradient descent. Robbins & Monro, 1951 ("A Stochastic
    # Approximation Method", Ann. Math. Statist. 22(3)).
    return step_size * grad
 
  if solver == 'sgdsign':
    # signSGD: Bernstein et al., 2018 ("signSGD: Compressed Optimisation
    # for Non-Convex Problems", ICML: arxiv.org/abs/1802.04434).
    return step_size * torch.sign(grad)
  
  if solver == 'adam':
    # Adam solver defined in the ZOO paper (Chen et al., 2017, "ZOO: Zeroth Order Optimization based
    # Black-box Attacks", AISec@CCS: arxiv.org/abs/1708.03999, doi:10.1145/3128572.3140448;
    # "Algorithm 2: ZOO-ADAM").
    if 'mt' not in state:
      state['mt']    = torch.zeros_like(grad)
      state['vt']    = torch.zeros_like(grad)
      state['epoch'] = 1                              # ZOO adam_epoch starts at 1
    state['mt'] = beta1 * state['mt'] + (1 - beta1) * grad
    state['vt'] = beta2 * state['vt'] + (1 - beta2) * grad * grad
    t    = state['epoch']
    corr = (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
    state['epoch'] += 1
    return step_size * corr * state['mt'] / (state['vt'].sqrt() + 1e-8)

  if solver == 'signum':
    # Signum: signSGD with momentum. Bernstein et al., 2018
    # ("signSGD with Majority Vote" / signSGD: arxiv.org/abs/1802.04434).
    if 'mt' not in state:
      state['mt'] = torch.zeros_like(grad)
    state['mt'] = beta1 * state['mt'] + (1 - beta1) * grad
    return step_size * torch.sign(state['mt'])

  if solver == 'lion':
    # Lion (EvoLved Sign Momentum: Chen et al., 2023 ("Symbolic
    # Discovery of Optimization Algorithms": arxiv.org/abs/2302.06675).
    if 'mt' not in state:
      state['mt'] = torch.zeros_like(grad)
    update = torch.sign(beta1 * state['mt'] + (1 - beta1) * grad)
    state['mt'] = beta2 * state['mt'] + (1 - beta2) * grad   # momentum updated after the step, like ZOO
    return step_size * update
 
  if solver == 'newton':
    # Newton's method: solver defined in the ZOO paper (Chen et al., 2017, "ZOO", AISec@CCS:
    # arxiv.org/abs/1708.03999, doi:10.1145/3128572.3140448; the "ZOO-Newton"
    # variant). Hessian diagonal via Hutchinson.
    if hvp_fn is None:
      raise ValueError("newton requires hvp_fn!")
    hess = hutchinson_diag(hvp_fn, grad)
    hess = torch.where(hess < 0, torch.ones_like(hess), hess)  # in zoo, neg: 1.0
    hess = torch.clamp(hess, min=HESS_FLOOR)                   # from ZOO implementation
    return step_size * grad / hess

  if solver == 'adahessian':
    # AdaHessian: Yao et al., 2021 ("ADAHESSIAN: An Adaptive Second Order
    # Optimizer for Machine Learning", AAAI:  arxiv.org/abs/2006.00719).
    if hvp_fn is None:
      raise ValueError("adahessian requires hvp_fn!")
    hess = hutchinson_diag(hvp_fn, grad)
    hess = torch.where(hess < 0, torch.ones_like(hess), hess)
    hess = torch.clamp(hess, min=HESS_FLOOR)
    if 'mt' not in state:
      state['mt']    = torch.zeros_like(grad)
      state['vt']    = torch.zeros_like(grad)
      state['epoch'] = 1
    state['mt'] = beta1 * state['mt'] + (1 - beta1) * grad
    state['vt'] = beta2 * state['vt'] + (1 - beta2) * hess * hess   # EMA of h^2
    t    = state['epoch']
    corr = (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
    state['epoch'] += 1
    return step_size * corr * state['mt'] / (state['vt'].sqrt() + 1e-8)
 
  raise ValueError("unknown solver %r: choose from available solvers: %s" % (solver, PGD_SOLVERS))
 

# Image helpers

def to_uint8(x_np):
    """(C,H,W) in [-0.5,0.5] -> (H,W,C) uint8 in [0,255]."""
    x = np.clip(x_np.transpose(1, 2, 0) + 0.5, 0.0, 1.0)
    return (x * 255).astype(np.uint8)

def save_image(arr_uint8, path):
    if arr_uint8.shape[2] == 1:
        Image.fromarray(arr_uint8[:, :, 0], mode='L').save(path)
    else:
        Image.fromarray(arr_uint8, mode='RGB').save(path)

def compute_metrics(orig_np, adv_np):
    """orig/adv are (C,H,W) in [-0.5,0.5]. Returns dict of MSE,MAE,PSNR,SSIM."""
    o = np.clip(orig_np.transpose(1,2,0)+0.5, 0, 1)
    a = np.clip(adv_np.transpose(1,2,0) +0.5, 0, 1)
    mse  = float(np.sum((o-a)**2))
    mae  = float(np.sum(np.abs(o-a)))
    psnr = float(calc_psnr(o, a, data_range=1.0))
    if o.shape[2] == 1:
        ssim = float(calc_ssim(o[:,:,0], a[:,:,0], data_range=1.0))
    else:
        ssim = float(calc_ssim(o, a, data_range=1.0, channel_axis=2))
    return {'mse': mse, 'mae': mae, 'psnr': psnr, 'ssim': ssim}

# -- main ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PGD White-Box Adversarial Attack')
    parser.add_argument('--dataset',  choices=['mnist','cifar10','imagenet'],
                        default='cifar10')
    parser.add_argument('--solver',   choices=PGD_SOLVERS, default='sgdsign', help='Optimizer that consumes the exact ' \
    'gradient. sgdsign (default) is classic PGD; the others let each optimizer be compared with the exact gradient ' \
    'here vs the estimated gradient in ZOO.')

    parser.add_argument('--targeted', action='store_true',
                        help='Targeted attack (default: untargeted)')
    parser.add_argument('--targeted-k', type=int, default=None,
                        help='Number of non-true target classes per source image in targeted mode '
                             '(default: all non-true classes)')
    parser.add_argument('--imagenette-one-per-class', action='store_true',
                        help='For ImageNet: use exactly one correctly-classified sample from each '
                             'ImageNette class (10 total sources).')
    parser.add_argument('--target-label-set', choices=['all', 'imagenette10'], default='all',
                        help='Target class pool for targeted attacks (default: all classes).')
    parser.add_argument('--targeted-classes', default=None,
                        help='Comma-separated class IDs to use as targets (targeted only). '
                             'If set, overrides --target-label-set.')
    parser.add_argument('--samples',  type=int, default=10)
    parser.add_argument('--start',    type=int, default=6,
                        help='Offset into the test set (same as ZOO / NES)')
    parser.add_argument('--imagenet_dir', default='./mini_imagenet')
    # Attack hyperparameters (None = auto per dataset)
    parser.add_argument('--epsilon',   type=float, default=None,
                        help='L-inf perturbation budget')
    parser.add_argument('--step_size', type=float, default=None,
                        help='PGD sign-step size (alpha)')
    parser.add_argument('--max_iter',  type=int,   default=None)
    parser.add_argument('--seed', type=int, default=42,
              help='Random seed for reproducible sample ordering')
    args = parser.parse_args()


    DEFAULTS = {
        'mnist':    dict(epsilon=0.3,  step_size=0.075,  max_iter=500),
        'cifar10':  dict(epsilon=0.05, step_size=0.0125, max_iter=500),
        'imagenet': dict(epsilon=0.05, step_size=0.0125, max_iter=200),
    }
    d = DEFAULTS[args.dataset]
    if args.epsilon   is None: args.epsilon   = d['epsilon']
    if args.step_size is None: args.step_size = d['step_size']
    if args.max_iter  is None: args.max_iter  = d['max_iter']

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, device_msg = choose_device_with_cuda_probe()
    print(device_msg)
    print('Dataset: %s | Attack: PGD (white-box) | Optimizer used as solver: %s | Targeted: %s | Device: %s' % (
        args.dataset, args.solver, args.targeted, device))

    # -- Load dataset + model -------------------------------------------------
    loader, model, num_labels, label_names = load_dataset_and_model(
      args.dataset, device, args.imagenet_dir, loader_seed=args.seed)

    # -- Filter to correctly-classified samples -------------------------------
    print('Checking model accuracy on test set...')
    data_correct, label_correct = [], []
    total = 0
    num_correct = 0
    with torch.no_grad():
      for img, lbl in loader:
        img_dev = img.to(device)
        lbl_int = int(lbl.item())
        pred = int(model(img_dev).argmax(dim=1).item())

        total += 1
        if pred == lbl_int:
          num_correct += 1
          data_correct.append(img[0].numpy())
          label_correct.append(lbl_int)

    acc = float(num_correct) / max(total, 1)
    print('Model accuracy on original samples: %.2f%%' % (acc * 100))

    data_correct = np.array(data_correct, dtype=np.float32)
    label_correct = np.array(label_correct, dtype=np.int64)
    print('Correctly classified: %d / %d' % (num_correct, total))

    correct_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(data_correct),
            torch.from_numpy(label_correct)),
        batch_size=1, shuffle=False)

    # -- Select attack samples ------------------------------------------------
    target_class_ids = None
    if args.targeted:
        if args.targeted_classes is not None:
            target_class_ids = [int(c) for c in args.targeted_classes.split(',')]
        elif args.target_label_set == 'imagenette10':
            target_class_ids = IMAGENETTE_LABEL_IDS

    use_imagenette_one_per_class = args.imagenette_one_per_class or (
      args.dataset == 'imagenet'
    )

    if use_imagenette_one_per_class:
        selected_data, selected_labels, missing = select_one_per_label(
            data_correct, label_correct, IMAGENETTE_LABEL_IDS, start=args.start)
        if missing:
            raise RuntimeError(
                'Could not find correctly-classified samples for labels: %s' % missing)

        if args.dataset == 'imagenet' and not args.imagenette_one_per_class:
          if args.targeted:
            print('Auto-enabling class-balanced ImageNette sources for targeted ImageNet.')
          else:
            print('Auto-enabling class-balanced ImageNette sources for untargeted ImageNet.')
        print('Using class-balanced ImageNette sources (10 classes, 1 sample each).')
        print('Selected labels: %s' % selected_labels.tolist())

        correct_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(selected_data),
                torch.from_numpy(selected_labels)),
            batch_size=1, shuffle=False)

        # We already curated exactly 10 sources, so consume all from the start.
        args.samples = len(selected_data)
        args.start = -1

    if args.dataset == 'imagenet' and args.targeted and args.targeted_k is None:
        args.targeted_k = 10
        print('Auto-setting targeted-k=10 for targeted ImageNet (10 target attacks per source image).')

    inputs, targets = generate_data(correct_loader, args.targeted,
                                    args.samples, args.start, num_labels,
                                    targeted_k=args.targeted_k,
                                    target_class_ids=target_class_ids)
    print('Attack samples selected: %d' % len(inputs))

    # -- Output directory -----------------------------------------------------
    # <dataset>/<targeted|untargeted>/pgd_<solver>/
    targeted_str = 'targeted' if args.targeted else 'untargeted'
    out_dir = os.path.join(args.dataset, targeted_str, 'pgd_' + args.solver)
    os.makedirs(out_dir, exist_ok=True)
    print("Data ready for imagenet")

    # -- Run attacks ----------------------------------------------------------
    adv_list      = []
    success_list  = []
    linf_list     = []
    l2_list       = []
    queries_list  = []
    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    orig_classes, adv_classes = [], []
    t0 = time.time()

    for i in range(len(inputs)):
        print('\n=== Sample %d / %d ===' % (i+1, len(inputs)))
        adv_np, success, dist_linf, dist_l2, _, queries = pgd_attack_one(
            inputs[i], targets[i], model, args.targeted, device,
            epsilon   = args.epsilon,
            step_size = args.step_size,
            max_iter  = args.max_iter,
            solver    = args.solver,
        )
        queries_list.append(queries)

        with torch.no_grad():
            oc = int(model(torch.from_numpy(inputs[i]).unsqueeze(0).to(device)
                           ).argmax(dim=1).item())
            ac = int(model(torch.from_numpy(adv_np).unsqueeze(0).to(device)
                           ).argmax(dim=1).item())
        orig_classes.append(oc)
        adv_classes.append(ac)

        save_image(to_uint8(inputs[i]), os.path.join(out_dir, 'original_%d.png'    % i))
        save_image(to_uint8(adv_np),    os.path.join(out_dir, 'adversarial_%d.png' % i))

        m = compute_metrics(inputs[i], adv_np)
        mse_list.append(m['mse']);  mae_list.append(m['mae'])
        psnr_list.append(m['psnr']); ssim_list.append(m['ssim'])

        adv_list.append(adv_np)
        success_list.append(bool(success))
        linf_list.append(dist_linf)
        l2_list.append(dist_l2)

        status = 'SUCCESS' if success else 'FAIL'
        on = label_names[oc][:12] if isinstance(label_names[oc], str) else str(oc)
        an = label_names[ac][:12] if isinstance(label_names[ac], str) else str(ac)
        print('[%s] orig:%s -> adv:%s | L-inf: %.4f | L2: %.4f | queries: %d' % (
            status, on, an, dist_linf, dist_l2, queries))

    elapsed = (time.time() - t0) / 60.0
    success_rate = 100.0 * sum(success_list) / max(len(success_list), 1)

    # total L2 distortion 
    total_distortion_l2 = float(np.sum(l2_list))

    successful_queries = [queries_list[i] for i in range(len(success_list))
                          if success_list[i]]
    mean_queries = float(np.mean(successful_queries)) if successful_queries \
                   else float('nan')

    # distortion averaged only over successful attacks
    succ_linf = [linf_list[i] for i in range(len(success_list)) if success_list[i]]
    succ_l2   = [l2_list[i]   for i in range(len(success_list)) if success_list[i]]
    mean_linf_on_success = float(np.mean(succ_linf)) if succ_linf else float('nan')
    mean_l2_on_success   = float(np.mean(succ_l2))   if succ_l2   else float('nan')

    print('\n' + '='*60)
    print('Attack : PGD (white-box, first-order baseline)')
    print('Optimizer used as solver : %s' % args.solver)
    print('Success Rate : %.1f %%' % success_rate)
    print('Queries (avg on success) : %.1f' % mean_queries)
    print('L-inf dist (avg on success): %.4f' % mean_linf_on_success)
    print('L2 dist    (avg on success): %.4f' % mean_l2_on_success)
    print('Time : %.2f mins for %d samples' % (elapsed, len(inputs)))
    print('='*60)

    # -- Save results.json ----------------------------------------------------
    results = {
        'dataset':                    args.dataset,
        'attack':                     'pgd',
        'targeted':                   args.targeted,
        'solver':                     args.solver,
        'num_samples':                len(inputs),
        'success_rate_pct':           success_rate,
        'total_distortion':           total_distortion_l2,   # L2 to match ZOO
        'time_mins':                  elapsed,
        'early_stop':                 True,
        'queries': {
            'per_sample':             queries_list,
            'mean_on_success':        mean_queries,
            'counting_convention':    '3 per PGD iter (1 fwd loss + 1 bwd grad + 1 fwd success-check)',
        },
        'distortion': {
            'linf_per_sample':        linf_list,
            'l2_per_sample':          l2_list,
            'mean_linf_on_success':   mean_linf_on_success,
            'mean_l2_on_success':     mean_l2_on_success,
        },
        'valid_classification':       orig_classes,
        'adversarial_classification': adv_classes,
        'mse':  {'mean': float(np.mean(mse_list)),  'per_sample': mse_list},
        'mae':  {'mean': float(np.mean(mae_list)),  'per_sample': mae_list},
        'psnr': {'mean': float(np.mean(psnr_list)), 'per_sample': psnr_list},
        'ssim': {'mean': float(np.mean(ssim_list)), 'per_sample': ssim_list},
        'attack_params': {
            'epsilon':   args.epsilon,
            'step_size': args.step_size,
            'max_iter':  args.max_iter,
        }
    }
    results_path = os.path.join(out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved to %s' % results_path)

    # -- grid visualisation ------------
    matplotlib.use('Agg')

    if args.dataset == 'imagenet':
        label_fn = lambda idx: (label_names[int(idx)][:12]
                                if isinstance(label_names[int(idx)], str)
                                else str(int(idx)))
    else:
        label_fn = lambda idx: label_names[int(idx)]

    cnt = 0
    plt.figure(figsize=(10, 10))
    for i in range(len(adv_list)):
        cnt += 1
        plt.subplot(10, 10, cnt)
        plt.xticks([], []); plt.yticks([], [])
        plt.title('%s\u2192%s' % (label_fn(orig_classes[i]),
                                  label_fn(adv_classes[i])), fontsize=5)
        img = np.clip((adv_list[i] + 0.5).transpose(1, 2, 0), 0, 1)
        if args.dataset == 'mnist':
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'grid_%s_%s_pgd_%s.png' % (args.dataset, targeted_str, args.solver))
    plt.savefig(grid_path, dpi=120)
    print('Grid saved to %s' % grid_path)