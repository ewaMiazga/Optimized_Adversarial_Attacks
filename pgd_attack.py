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
from setup_imagenet_model import VGG16Wrapper, imagenet_transform, get_imagenet_labels

## copying zoo l2 attack dataset handling, images normalized to [-0.5, 0.5]
def load_dataset_and_model(dataset_name, device, imagenet_dir=None):

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
        if imagenet_dir is None:
            raise ValueError('--imagenet_dir is required when --dataset imagenet')
        from torchvision.datasets import ImageFolder
        test_set = ImageFolder(root=imagenet_dir, transform=imagenet_transform())
        model = VGG16Wrapper(pretrained=True).to(device)
        num_labels  = 1000
        label_names = get_imagenet_labels()

    else:
        raise ValueError('Unknown dataset: %s' % dataset_name)

    model.eval()
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    return loader, model, num_labels, label_names


## mirror generate_data of zoo l2 attack
def generate_data(loader, targeted, samples, start, num_labels):
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
            for j in range(num_labels):
                if j == lbl:
                    continue
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
    parser.add_argument('--samples',  type=int, default=10)
    parser.add_argument('--start',    type=int, default=6,
                        help='Offset into the test set (same as ZOO / NES)')
    parser.add_argument('--imagenet_dir', default=None,
                        help='Path to ImageNet val directory (ImageFolder layout)')
    # Attack hyperparameters (None = auto per dataset)
    parser.add_argument('--epsilon',   type=float, default=None,
                        help='L-inf perturbation budget')
    parser.add_argument('--step_size', type=float, default=None,
                        help='PGD sign-step size (alpha)')
    parser.add_argument('--max_iter',  type=int,   default=None)
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

    np.random.seed(42)
    torch.manual_seed(42)

    use_cuda = True
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available())
                           else 'cpu')
    print('Dataset: %s | Attack: PGD (white-box) | Optimizer used as solver: %s | Targeted: %s | Device: %s' % (
        args.dataset, args.solver, args.targeted, device))

    # -- Load dataset + model -------------------------------------------------
    loader, model, num_labels, label_names = load_dataset_and_model(
        args.dataset, device, args.imagenet_dir)

    # -- Filter to correctly-classified samples -------------------------------
    print('Checking model accuracy on test set...')
    all_inputs, all_labels = [], []
    for img, lbl in loader:
        all_inputs.append(img[0].numpy())
        all_labels.append(int(lbl.item()))
    all_inputs = np.array(all_inputs)
    all_labels = np.array(all_labels)

    inp_t = torch.from_numpy(all_inputs).to(device)
    with torch.no_grad():
        preds = model(inp_t).argmax(dim=1).cpu().numpy()

    acc = (preds == all_labels).mean()
    print('Model accuracy on original samples: %.2f%%' % (acc * 100))

    correct_mask  = (preds == all_labels)
    data_correct  = all_inputs[correct_mask]
    label_correct = all_labels[correct_mask]
    print('Correctly classified: %d / %d' % (correct_mask.sum(), len(all_labels)))

    correct_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(data_correct),
            torch.from_numpy(label_correct)),
        batch_size=1, shuffle=False)

    # -- Select attack samples ------------------------------------------------
    inputs, targets = generate_data(correct_loader, args.targeted,
                                    args.samples, args.start, num_labels)
    print('Attack samples selected: %d' % len(inputs))

    # -- Output directory -----------------------------------------------------
    # <dataset>/<targeted|untargeted>/pgd_<solver>/
    targeted_str = 'targeted' if args.targeted else 'untargeted'
    out_dir = os.path.join(args.dataset, targeted_str, 'pgd_' + args.solver)
    os.makedirs(out_dir, exist_ok=True)

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