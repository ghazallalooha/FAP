import numpy as np, torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import norm
from fap.data.cmmd_dataset import get_cmmd_dataloader
from fap.models.rblur_vgg import get_model

@torch.no_grad()
def evaluate_clean(data_root, batch_size=16, num_fixations=3, ckpt=None, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model  = get_model(num_classes=2, num_fixations=num_fixations).to(device)
    if ckpt: model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    loader = get_cmmd_dataloader(data_root, 'test', batch_size, num_fixations, device='cpu', shuffle=False)
    correct=total=0
    for xk, y in loader:
        x = xk.view(-1,3,224,224).float().to(device); y = y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    acc = correct/total
    print(f"[CMMD Clean] Accuracy: {acc*100:.2f}%"); return acc

@torch.no_grad()
def certify_randomized_smoothing(data_root, batch_size=1, num_fixations=3, ckpt=None,
                                 sigma=0.12, n_samples=10000, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model  = get_model(num_classes=2, num_fixations=num_fixations).to(device)
    if ckpt: model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    loader = get_cmmd_dataloader(data_root, 'test', batch_size, num_fixations, device='cpu', shuffle=False)
    radii, correct, total = [], 0, 0
    for xk, y in tqdm(loader):
        x = xk.view(-1,3,224,224).float().to(device); y = y.to(device)
        x = x.view(-1, num_fixations, 3, 224, 224).mean(dim=1)
        counts = torch.zeros(2, device=device)
        for _ in range(n_samples):
            noise = sigma * torch.randn_like(x)
            counts[model((x + noise).clamp(0,1)).argmax(1).item()] += 1
        pA = counts.max().item()/n_samples; pred = counts.argmax().item()
        radius = sigma * norm.ppf(pA) if pA > 0.5 else 0.0
        radii.append(radius); correct += int(pred == y.item()); total += 1
    print(f"[CMMD Cert] mean r={np.mean(radii):.4f}, acc={correct/total:.4f}")
    return float(np.mean(radii))
