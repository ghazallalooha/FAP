import torch
import torch.nn as nn

class MaskedPGD:
    """
    δ_{t+1} = Proj_ε( δ_t + α * sign(∇_x L) ⊙ M )
    M=1 outside lesion ROI; M=0 inside ROI → lesion-aware robustness.
    """
    def __init__(self, model, eps=0.004, alpha=0.001, steps=5):
        self.model = model
        self.eps, self.alpha, self.steps = eps, alpha, steps
        self.crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _project(self, x_nat, x_cur):
        x_cur = torch.clamp(x_cur, 0, 1)
        delta = torch.clamp(x_cur - x_nat, -self.eps, self.eps)
        return torch.clamp(x_nat + delta, 0, 1)

    def perturb(self, x_nat, y, mask=None):
        x = x_nat.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            out = self.model(x)
            loss = self.crit(out, y)
            g = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
            step = self.alpha * torch.sign(g)
            if mask is not None:
                step = step * mask
            with torch.no_grad():
                x = self._project(x_nat, x + step).detach().requires_grad_(True)
        return x.detach()
