import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import math
import time
import os
import sys
import json
import argparse
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from setup_mnist_model import MNIST
from setup_cifar10_model import CIFAR10
from setup_imagenet_model import VGG16Wrapper, imagenet_transform, get_imagenet_labels

"""##L2 Black Box Attack"""

# All solvers operate entirely on GPU tensors — no CPU round-trips.
# Signature: (losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj)
#   losses       : 1-D GPU float tensor, shape (2*batch_size+1,)  — [cur, +d0, -d0, +d1, -d1, ...]
#   indice       : 1-D GPU long tensor,  shape (batch_size,)       — pixel indices being updated
#   mt, vt       : 1-D GPU float tensors, shape (var_len,)         — momentum buffers
#   adam_epoch   : 1-D GPU float tensor, shape (var_len,)          — per-coord step counter
#   real_modifier: GPU float tensor, shape (1, C, H, W)            — perturbation (modified in-place)
#   up, down     : 1-D GPU float tensors, shape (var_len,)         — projection bounds

def coordinate_ADAM(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  g = ((losses[1::2] - losses[2::2]) / 0.0002).float()  # cast to float32 to match state tensors
  mt[indice] = beta1 * mt[indice] + (1 - beta1) * g
  vt[indice] = beta2 * vt[indice] + (1 - beta2) * g * g
  epoch = adam_epoch[indice]
  corr = torch.sqrt(1 - beta2 ** epoch) / (1 - beta1 ** epoch)
  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * corr * mt[indice] / (torch.sqrt(vt[indice]) + 1e-8)
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])
  adam_epoch[indice] += 1

def coordinate_Newton(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  cur_loss = losses[0]
  g    = ((losses[1::2] - losses[2::2]) / 0.0002).float()
  hess = ((losses[1::2] - 2 * cur_loss + losses[2::2]) / (0.0001 ** 2)).float()
  hess = torch.where(hess < 0, torch.ones_like(hess), hess)  # negative hess → 1.0
  hess = torch.clamp(hess, min=0.1)                           # hess < 0.1 → 0.1
  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * g / hess
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])

def coordinate_SGD(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  """Vanilla gradient descent — no momentum."""
  g = ((losses[1::2] - losses[2::2]) / 0.0002).float()
  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * g
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])

def coordinate_SGDSign(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  """SGDSign — step is step_size * sign(g), no momentum."""
  g = ((losses[1::2] - losses[2::2]) / 0.0002).float()
  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * torch.sign(g)
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])

def coordinate_Signum(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  """Signum — step is step_size * sign(m), where m is an EMA of gradients."""
  g = ((losses[1::2] - losses[2::2]) / 0.0002).float()
  mt[indice] = beta1 * mt[indice] + (1 - beta1) * g
  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * torch.sign(mt[indice])
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])

def coordinate_Lion(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  """Lion optimizer (EvoLved Sign Momentum) — arxiv.org/abs/2302.06675
  u = sign(beta1*m + (1-beta1)*g)  ->  theta -= lr*u  ->  m = beta2*m + (1-beta2)*g
  """
  g = ((losses[1::2] - losses[2::2]) / 0.0002).float()
  update = torch.sign(beta1 * mt[indice] + (1 - beta1) * g)  # direction from interpolated momentum
  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * update
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])
  mt[indice] = beta2 * mt[indice] + (1 - beta2) * g          # momentum updated AFTER the step

def coordinate_AdaHessian(losses, indice, mt, vt, adam_epoch, real_modifier, up, down, step_size, beta1, beta2, proj):
  """AdaHessian (Yao et al., 2021) — arxiv.org/abs/2006.00719
  Second moment tracks EMA of h², where h is the FD Hessian diagonal.
  Negative-curvature coordinates fall back to h=1 (same as Newton) to avoid
  freezing in saddle regions. vt is initialised effectively at hess_floor² to
  avoid the cold-start blow-up that occurs when vt≈0 at the first iterations.

  mt : EMA of gradient               (first moment)
  vt : EMA of h²                     (second moment — Hessian-based, not gradient-based)
  Update: theta -= lr * m_hat / (sqrt(v_hat) + eps)
  """
  HESS_FLOOR = 0.1   # match Newton's floor so saddle regions behave consistently
  cur_loss = losses[0]
  g    = ((losses[1::2] - losses[2::2]) / 0.0002).float()
  hess = ((losses[1::2] - 2 * cur_loss + losses[2::2]) / (0.0001 ** 2)).float()
  # Negative curvature → fall back to 1.0, same policy as Newton
  hess = torch.where(hess < 0, torch.ones_like(hess), hess)
  hess = torch.clamp(hess, min=HESS_FLOOR)

  mt[indice] = beta1 * mt[indice] + (1 - beta1) * g
  vt[indice] = beta2 * vt[indice] + (1 - beta2) * hess * hess   # EMA of h²

  epoch = adam_epoch[indice]
  corr  = torch.sqrt(1 - beta2 ** epoch) / (1 - beta1 ** epoch)

  m = real_modifier.view(-1)
  m[indice] = m[indice] - step_size * corr * mt[indice] / (torch.sqrt(vt[indice]) + 1e-8)
  if proj:
    m[indice] = torch.clamp(m[indice], down[indice], up[indice])
  adam_epoch[indice] += 1

def loss_run(input,target,model,modifier,use_tanh,use_log,targeted,confidence,const,device='cpu'):
  if use_tanh:
    pert_out = torch.tanh(input +modifier)/2
  else:
    pert_out = input + modifier

  output = model(pert_out)
  if use_log:
    output = F.softmax(output,-1)
  
  # l2 distance 
  if use_tanh:
    loss1 = torch.sum(torch.square(pert_out-torch.tanh(input)/2),dim=(1,2,3))
  else:
    loss1 = torch.sum(torch.square(pert_out-input),dim=(1,2,3))
  
  # l1 distance
  # if use_tanh:
  #   loss1 = torch.sum(torch.abs(pert_out-torch.tanh(input)/2),dim=(1,2,3))
  # else:
  #   loss1 = torch.sum(torch.abs(pert_out-input),dim=(1,2,3))

  # l inf distance
  # if use_tanh:
  #   loss1 = torch.max(torch.abs(pert_out-torch.tanh(input)/2).view(pert_out.size(0),-1),dim=-1)[0]
  # else:
  #   loss1 = torch.max(torch.abs(pert_out-input).view(pert_out.size(0),-1),dim=-1)[0]

  real = torch.sum(target*output,-1)
  other = torch.max((1-target)*output-(target*10000),-1)[0]
 
  if use_log:
    real=torch.log(real+1e-30)
    other=torch.log(other+1e-30)
  
  confidence = torch.tensor(confidence).type(torch.float64).to(device)
  
  if targeted:
    loss2 = torch.max(other-real,confidence)
  else:
    loss2 = torch.max(real-other,confidence)
  
  loss2 = const*loss2
  l2 = loss1
  loss = loss1 + loss2
  
  # losses/l2/loss2 stay on GPU (consumed by solvers); scores+pert_images go to CPU for output
  return loss.detach(), l2.detach(), loss2.detach(), output.detach().cpu().numpy(), pert_out.detach().cpu().numpy()

def l2_attack(input, target, model, targeted, use_log, use_tanh, solver, device='cpu', reset_adam_after_found=True,abort_early=True,
              batch_size=128,max_iter=1000,const=0.01,confidence=0.0,early_stop_iters=100, binary_search_steps=9,
              step_size=0.01,adam_beta1=0.9,adam_beta2=0.999):
  
  early_stop_iters = early_stop_iters if early_stop_iters != 0 else max_iter // 10

  input = torch.from_numpy(input).to(device)
  target = torch.from_numpy(target).to(device)
  
  var_len = input.view(-1).size()[0]
  # All state tensors live on device — no CPU↔GPU copies during the attack loop
  modifier_up   = torch.zeros(var_len, dtype=torch.float32, device=device)
  modifier_down = torch.zeros(var_len, dtype=torch.float32, device=device)
  real_modifier = torch.zeros(input.size(), dtype=torch.float32, device=device)
  mt            = torch.zeros(var_len, dtype=torch.float32, device=device)
  vt            = torch.zeros(var_len, dtype=torch.float32, device=device)
  adam_epoch    = torch.ones(var_len,  dtype=torch.float32, device=device)

  upper_bound=1e10
  lower_bound=0.0
  out_best_attack=input.clone().detach().cpu().numpy()
  out_best_const=const  
  out_bestl2=1e10
  out_bestscore=-1
  

  if use_tanh:
    input = torch.atanh(input*1.99999)

  if not use_tanh:
    flat = input.clone().detach().view(-1)
    modifier_up   =  0.5 - flat
    modifier_down = -0.5 - flat
  
  def compare(x,y):
    if not isinstance(x, (float, int, np.int64)):
      if targeted:
        x[y] -= confidence
      else:
        x[y] += confidence
      x = np.argmax(x)
    if targeted:
      return x == y
    else:
      return x != y

  for step in range(binary_search_steps):
    bestl2 = 1e10
    prev=1e6
    bestscore=-1
    last_loss2=1.0
    # reset solver state
    mt.zero_()
    vt.zero_()
    adam_epoch.fill_(1)
    stage=0
    
    for iter in range(max_iter):
      if (iter+1)%100 == 0:
        loss, l2, loss2, _ , __ = loss_run(input,target,model,real_modifier,use_tanh,use_log,targeted,confidence,const,device)
        print("[STATS][L2] iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter+1, loss[0].item(), l2[0].item(), loss2[0].item()))
        sys.stdout.flush()

      # Sample random coordinates entirely on GPU — no numpy, no CPU transfer
      indice = torch.randperm(var_len, device=device)[:batch_size]
      # Build (2*batch_size+1) perturbed copies of real_modifier on GPU
      var = real_modifier.expand(batch_size * 2 + 1, *input.shape[1:]).clone()
      flat_var = var.view(batch_size * 2 + 1, -1)
      pos_idx = torch.arange(batch_size, device=device) * 2 + 1
      neg_idx = torch.arange(batch_size, device=device) * 2 + 2
      flat_var[pos_idx, indice] += 0.0001
      flat_var[neg_idx, indice] -= 0.0001

      losses, l2s, losses2, scores, pert_images = loss_run(input,target,model,var,use_tanh,use_log,targeted,confidence,const,device)

      # Solver updates real_modifier in-place — everything stays on GPU
      if solver=="adam":
        coordinate_ADAM(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
      if solver=="newton":
        coordinate_Newton(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
      if solver=="sgd":
        coordinate_SGD(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
      if solver=="sgdsign":
        coordinate_SGDSign(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
      if solver=="signum":
        coordinate_Signum(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
      if solver=="lion":
        coordinate_Lion(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)
      if solver=="adahessian":
        coordinate_AdaHessian(losses,indice,mt,vt,adam_epoch,real_modifier,modifier_up,modifier_down,step_size,adam_beta1,adam_beta2,proj=not use_tanh)

      loss2_val = losses2[0].item()
      if loss2_val==0.0 and last_loss2!=0.0 and stage==0:
        if reset_adam_after_found:
          mt.zero_()
          vt.zero_()
          adam_epoch.fill_(1)
        stage=1
      last_loss2=loss2_val

      loss_val = losses[0].item()
      if abort_early and (iter+1) % early_stop_iters == 0:
        if loss_val > prev*.9999:
            print("Early stopping because there is no improvement")
            break
        prev = loss_val

      l2_val = l2s[0].item()
      target_label = np.argmax(target.detach().cpu().numpy(),-1)
      if l2_val < bestl2 and compare(scores[0], target_label):
        bestl2 = l2_val
        bestscore = np.argmax(scores[0])

      if l2_val < out_bestl2 and compare(scores[0], target_label):
        if out_bestl2 == 1e10:
          print("[STATS][L3](First valid attack found!) iter = {}, loss = {:.5f}, loss1 = {:.5f}, loss2 = {:.5f}".format(iter+1, loss_val, l2_val, loss2_val))
          sys.stdout.flush()
        out_bestl2 = l2_val
        out_bestscore = np.argmax(scores[0])
        out_best_attack = pert_images[0]
        out_best_const = const
  
    if compare(bestscore, np.argmax(target.detach().cpu().numpy(),-1)) and bestscore != -1:
      print('old constant: ', const)
      upper_bound = min(upper_bound,const)
      if upper_bound < 1e9:
          const = (lower_bound + upper_bound)/2
      print('new constant: ', const)
    else:
      print('old constant: ', const)
      lower_bound = max(lower_bound,const)
      if upper_bound < 1e9:
          const = (lower_bound + upper_bound)/2
      else:
          const *= 10
      print('new constant: ', const)

  return out_best_attack, out_bestscore

def generate_data(test_loader, targeted, samples, start, num_label=10):
  inputs=[]
  targets=[]
  cnt=0
  for i, data in enumerate(test_loader):
    if cnt<samples:
      if i>start:
        data, label = data[0],data[1]
        if targeted:
          seq = range(num_label)
          for j in seq:
            if j==label.item():
              continue
            inputs.append(data[0].numpy())
            targets.append(np.eye(num_label)[j])
        else:
          inputs.append(data[0].numpy())
          targets.append(np.eye(num_label)[label.item()])
        cnt+=1
      else:
        continue
    else:
      break

  inputs=np.array(inputs)
  targets=np.array(targets)

  return inputs,targets

def attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device, step_size=0.01):
  r = []
  print('go up to',len(inputs))
  # run 1 image at a time, minibatches used for gradient evaluation
  for i in range(len(inputs)):
    print('tick',i+1)
    attack,score=l2_attack(np.expand_dims(inputs[i],0), np.expand_dims(targets[i],0), model, targeted, use_log, use_tanh, solver, device=device, step_size=step_size)
    r.append(attack)
  return np.array(r)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description="ZOO L2 Black-Box Adversarial Attack")
  parser.add_argument("--dataset",  choices=["mnist", "cifar10", "imagenet"], default="cifar10",
                      help="Dataset to attack (default: cifar10)")
  parser.add_argument("--solver",   choices=["adam", "newton", "sgd", "sgdsign", "signum", "lion", "adahessian"],
                      default="newton", help="Coordinate-descent solver (default: newton)")
  parser.add_argument("--targeted", action="store_true",
                      help="Run a targeted attack (default: untargeted)")
  parser.add_argument("--samples",  type=int, default=10,
                      help="Number of samples to attack (default: 10)")
  parser.add_argument("--start",    type=int, default=6,
                      help="Offset into the test set (default: 6)")
  parser.add_argument("--imagenet_dir", default=None,
                      help="Path to ImageNet val directory (required when --dataset imagenet)")
  args = parser.parse_args()

  np.random.seed(42)
  torch.manual_seed(42)

  dataset_name = args.dataset
  solver       = args.solver
  targeted     = args.targeted

  use_cuda = True
  device   = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
  print(f"Dataset: {dataset_name} | Solver: {solver} | Targeted: {targeted} | Device: {device}")

  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
  if dataset_name == "mnist":
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    model = MNIST().to(device)
    model.load_state_dict(torch.load('./models/mnist_model.pt', map_location=device, weights_only=False))
    num_label = 10
  elif dataset_name == "imagenet":
    if args.imagenet_dir is None:
      raise ValueError("--imagenet_dir is required when --dataset imagenet")
    from torchvision.datasets import ImageFolder
    test_set = ImageFolder(root=args.imagenet_dir, transform=imagenet_transform())
    model = VGG16Wrapper(pretrained=True).to(device)
    num_label = 1000
  else:
    test_set = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    model = CIFAR10().to(device)
    model.load_state_dict(torch.load('./models/cifar10_model.pt', map_location=device, weights_only=False))
    num_label = 10
  model.eval()

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

  use_log  = True
  use_tanh = True

  # ── Per-optimizer learning rates ─────────────────────────────────────────
  ADAM_LR    = 0.01
  NEWTON_LR  = 0.01
  SGD_LR     = 0.01
  SGDSIGN_LR = 0.001
  SIGNUM_LR  = 0.001
  LION_LR       = 0.001
  ADAHESSIAN_LR = 0.01

  lr_map = {
      "adam":       ADAM_LR,
      "newton":     NEWTON_LR,
      "sgd":        SGD_LR,
      "sgdsign":    SGDSIGN_LR,
      "signum":     SIGNUM_LR,
      "lion":       LION_LR,
      "adahessian": ADAHESSIAN_LR,
  }
  step_size = lr_map[solver]
  # ─────────────────────────────────────────────────────────────────────────

  #start is a offset to start taking sample from test set
  #samples is the how many samples to take in total : for targeted, 1 means all 9 class target -> 9 total samples whereas for untargeted the original data 
  #sample is taken i.e. 1 sample only 
  
  # check the accuracy of the model on the orginal samples 
  data, label = generate_data(test_loader,targeted=False,samples=len(test_loader),start=args.start,num_label=num_label)
  pred = model(torch.from_numpy(data).to(device))
  pred = torch.argmax(pred,dim=-1).cpu().numpy()
  label = np.argmax(label,axis=-1)
  acc = (pred==label).sum()/len(pred)
  print("Model accuracy on original samples: ", acc*100.0, "%")
  # per class classification accuracy
  print("labels: ", label)
  for i in range(10):
    idx = label==i
    acc = (pred[idx]==label[idx]).sum()/idx.sum()
    print("Class {} accuracy: {:.2f}%".format(i, acc*100.0))

  # exclude the missclassifed samples from the attack evaluation
  data = data[pred==label]
  label = label[pred==label]
  print("Number of correctly classified samples: ", len(data))
  test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label)), batch_size=1, shuffle=False)

  inputs, targets = generate_data(test_loader,targeted,samples=args.samples,start=args.start,num_label=num_label)
  timestart = time.time()
  adv = attack(inputs, targets, model, targeted, use_log, use_tanh, solver, device, step_size=step_size)
  timeend = time.time()
  print("Took",(timeend-timestart)/60.0,"mins to run",len(inputs),"samples.")

  if use_log:
    valid_class = np.argmax(F.softmax(model(torch.from_numpy(inputs).to(device)), -1).detach().cpu().numpy(), -1)
    adv_class   = np.argmax(F.softmax(model(torch.from_numpy(adv).to(device)), -1).detach().cpu().numpy(), -1)
  else:
    valid_class = np.argmax(model(torch.from_numpy(inputs).to(device)).detach().cpu().numpy(), -1)
    adv_class   = np.argmax(model(torch.from_numpy(adv).to(device)).detach().cpu().numpy(), -1)

  acc              = ((valid_class == adv_class).sum()) / len(inputs)
  success_rate     = (1.0 - acc) * 100.0
  # total change depends on input size (L2)
  total_distortion = float(np.sum((adv - inputs) ** 2) ** 0.5)
  elapsed_mins     = (timeend - timestart) / 60.0

  print("Valid Classification:      ", valid_class)
  print("Adversarial Classification:", adv_class)
  print("Success Rate:              ", success_rate, "%")
  print("Total distortion:          ", total_distortion)

  # ── Output directory: <dataset>/<targeted|untargeted>/<solver>/ ──────────────
  targeted_str = "targeted" if targeted else "untargeted"
  out_dir = os.path.join(dataset_name, targeted_str, solver)
  os.makedirs(out_dir, exist_ok=True)

  # ── Save images and compute perceptual metrics ────────────────────────────────
  # Inputs are normalised with Normalize((0.5,),(1.0,)) → pixel = value + 0.5
  mse_list = []
  mae_list = []
  psnr_list = []
  ssim_list = []

  for i in range(len(inputs)):
    orig_np = np.clip(inputs[i].transpose(1, 2, 0) + 0.5, 0.0, 1.0)  # (H,W,C) in [0,1]
    adv_np  = np.clip(adv[i].transpose(1, 2, 0)   + 0.5, 0.0, 1.0)

    orig_uint8 = (orig_np * 255).astype(np.uint8)
    adv_uint8  = (adv_np  * 255).astype(np.uint8)

    if orig_uint8.shape[2] == 1:          # grayscale (MNIST)
      orig_img = Image.fromarray(orig_uint8[:, :, 0], mode="L")
      adv_img  = Image.fromarray(adv_uint8[:, :, 0],  mode="L")
    else:                                  # RGB (CIFAR-10)
      orig_img = Image.fromarray(orig_uint8, mode="RGB")
      adv_img  = Image.fromarray(adv_uint8,  mode="RGB")

    orig_img.save(os.path.join(out_dir, f"original_{i}.png"))
    adv_img.save(os.path.join(out_dir,  f"adversarial_{i}.png"))

    mse = float(np.sum((orig_np - adv_np)**2))
    mae = float(np.sum(np.abs(orig_np - adv_np)))
    psnr = float(calc_psnr(orig_np, adv_np, data_range=1.0))
    if orig_np.shape[2] == 1:
      ssim = float(calc_ssim(orig_np[:, :, 0], adv_np[:, :, 0], data_range=1.0))
    else:
      ssim = float(calc_ssim(orig_np, adv_np, data_range=1.0, channel_axis=2))

    mse_list.append(mse)
    mae_list.append(mae)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

  # ── Write results.json ────────────────────────────────────────────────────────
  results = {
      "dataset":                    dataset_name,
      "targeted":                   targeted,
      "solver":                     solver,
      "num_samples":                len(inputs),
      "success_rate_pct":           success_rate,
      "total_distortion":           total_distortion,
      "time_mins":                  elapsed_mins,
      "valid_classification":       valid_class.tolist(),
      "adversarial_classification": adv_class.tolist(),
      "mse": {
          "mean:"       : float(np.mean(mse_list)),
          "per_sample":  mse_list
      },
      "mae": {
          "mean":       float(np.mean(mae_list)),
          "per_sample": mae_list
      },
      "psnr": {
          "mean":       float(np.mean(psnr_list)),
          "per_sample": psnr_list
      },
      "ssim": {
          "mean":       float(np.mean(ssim_list)),
          "per_sample": ssim_list
      }
  }

  results_path = os.path.join(out_dir, "results.json")
  with open(results_path, "w") as f:
      json.dump(results, f, indent=2)
  print(f"Results saved to {results_path}")

  # plt.tight_layout()
  # if targeted:
  #   if solver=="newton":
  #     plt.savefig('newton_targeted_mnist.png')
  #   else:
  #     plt.savefig('adam_targeted_mnist.png') 
  # else:
  #   if solver=="newton":
    #   plt.savefig('newton_untargeted_mnist.png')
    # else:
    #   plt.savefig('adam_untargeted_mnist.png') 

  #visualization of adversarial examples
  if dataset_name == "imagenet":
    imagenet_labels = get_imagenet_labels()
    label_fn = lambda idx: imagenet_labels[int(idx)][:12]
  elif dataset_name == "cifar10":
    cifar_classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    label_fn = lambda idx: cifar_classes[int(idx)]
  else:
    label_fn = lambda idx: str(int(idx))

  cnt=0
  plt.figure(figsize=(10,10))
  for i in range(len(adv)):
    cnt+=1
    plt.subplot(10,10,cnt)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title(f"{label_fn(valid_class[i])}\u2192{label_fn(adv_class[i])}", fontsize=5)
    img = np.clip((adv[i]+0.5).transpose(1,2,0), 0, 1)
    if dataset_name == "mnist":
      plt.imshow(img.squeeze(), cmap="gray")
    else:
      plt.imshow(img)
  plt.tight_layout()
  targeted_str2 = "targeted" if targeted else "untargeted"
  plt.savefig(os.path.join(out_dir, f"grid_{dataset_name}_{targeted_str2}_{solver}.png"), dpi=120)
