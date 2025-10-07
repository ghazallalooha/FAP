import os, torch
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from fap.data.cmmd_dataset import get_cmmd_dataloader
from fap.models.rblur_vgg import get_model
from fap.fap_core.fixation import gradient_guided_fixations
from fap.fap_core.fap_blur import FAPBlur
from fap.fap_core.masks import upsample_cam
from fap.fap_core.attacks import MaskedPGD

def train(data_root, out_dir="checkpoints", epochs=20, batch_size=32, num_fixations=3, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2, num_fixations=num_fixations).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler('cuda', enabled=(device.type=='cuda'))

    tr = get_cmmd_dataloader(data_root, 'train', batch_size, num_fixations)
    va = get_cmmd_dataloader(data_root, 'val',   batch_size, num_fixations)

    fap_blur = FAPBlur().to(device)
    attacker = MaskedPGD(model, eps=0.004, alpha=0.001, steps=5)

    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(tr, desc=f"epoch {ep}/{epochs}")
        for xk, y in pbar:
            xk, y = xk.to(device), y.to(device)
            fixs = gradient_guided_fixations(model, xk[:,0], y, F.cross_entropy, K=num_fixations, gamma=0.1)
            x_fap = fap_blur(xk[:,0], fixs)
            logits_clean = model(x_fap)
            cams_small = model.cams(logits_clean)
            M = 1.0 - (upsample_cam(cams_small, x_fap.size(2), x_fap.size(3)) >= 0.5).float()
            x_adv = attacker.perturb(x_fap, y, mask=M)

            opt.zero_grad(set_to_none=True)
            with autocast(device.type):
                loss = 0.7*F.cross_entropy(model(x_fap), y) + 0.3*F.cross_entropy(model(x_adv), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        acc = _eval(model, va, device)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "vgg16_fap_cmmd.pt"))
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "vgg16_fap_cmmd_best.pt"))
        print(f"[CMMD val] acc={acc*100:.2f}% (best {best*100:.2f}%)")

def _eval(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for xk,y in loader:
            x = xk[:,0].to(device); y=y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/total

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fix", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.data_root, args.out, args.epochs, args.batch, args.fix, args.lr)
