import os, torch
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from fap.data.breast_dataset import get_breast_loader
from fap.models.rblur_vgg import get_model
from fap.fap_core.attacks import MaskedPGD

def train(images_dir, labels_csv, out_dir="checkpoints_breast", epochs=20, batch_size=32, num_fixations=3, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2, num_fixations=num_fixations).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = GradScaler('cuda', enabled=(device.type=='cuda'))
    tr = get_breast_loader(images_dir, labels_csv, batch_size, num_fixations)
    best = 0.0
    attacker = MaskedPGD(model, eps=0.004, alpha=0.001, steps=5)

    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(tr, desc=f"epoch {ep}/{epochs}")
        for xk,y in pbar:
            x = xk[:,0].to(device); y=y.to(device)  # use one fixation per step for speed
            x_adv = attacker.perturb(x, y, mask=None)  # ROI masks optional for generic BREAST
            opt.zero_grad(set_to_none=True)
            with autocast(device.type):
                loss = 0.7*F.cross_entropy(model(x), y) + 0.3*F.cross_entropy(model(x_adv), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        # quick holdout-style eval on the tail of sampler (optional)
        acc = 0.0
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "vgg16_fap_breast.pt"))
        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "vgg16_fap_breast_best.pt"))
        print(f"[BREAST] epoch {ep} done")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="checkpoints_breast")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fix", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.images, args.csv, args.out, args.epochs, args.batch, args.fix, args.lr)
