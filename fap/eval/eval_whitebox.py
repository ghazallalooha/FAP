import torch, torch.nn.functional as F
from tqdm import tqdm
from fap.data.cmmd_dataset import get_cmmd_dataloader
from fap.models.rblur_vgg import get_model
from fap.fap_core.attacks import MaskedPGD
from fap.fap_core.masks import upsample_cam

@torch.no_grad()
def robust_accuracy_pgd(data_root, ckpt, eps=0.004, alpha=0.001, steps=25, num_fixations=3, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2, num_fixations=num_fixations).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)); model.eval()
    loader = get_cmmd_dataloader(data_root, 'test', batch_size=8, num_fixations=num_fixations, device='cpu', shuffle=False)
    attacker = MaskedPGD(model, eps=eps, alpha=alpha, steps=steps)
    correct = total = 0
    for xk, y in tqdm(loader):
        x = xk[:,0].to(device); y = y.to(device)
        logits = model(x)
        cams_small = model.cams(logits)
        M = 1.0 - (upsample_cam(cams_small, x.size(2), x.size(3)) >= 0.5).float()
        x_adv = attacker.perturb(x, y, mask=M)
        pred = model(x_adv).argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    acc = correct/total
    print(f"[White-box PGD] Robust accuracy: {acc*100:.2f}%")
    return acc
