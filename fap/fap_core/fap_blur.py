import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

class FAPBlur(nn.Module):
    """
    Eccentricity-adaptive separable Gaussian blur:
      sigma(d) = beta * sigmoid(alpha * (d_mm - mu_lesion_mm))
    """
    def __init__(self, beta=0.05, alpha=1.5, mu_lesion=18.2, pixel_mm=0.07, max_sigma_px=12, quant_bins=16):
        super().__init__()
        self.beta = beta; self.alpha = alpha; self.mu = mu_lesion
        self.pixel_mm = pixel_mm; self.max_sigma_px = max_sigma_px; self.quant_bins = quant_bins

    @torch.no_grad()
    def forward(self, x, fixations):
        device = x.device
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        yy = yy[None, None].repeat(B, 1, 1, 1); xx = xx[None, None].repeat(B, 1, 1, 1)
        d = torch.full((B,1,H,W), float('inf'), device=device)
        for b in range(B):
            if not fixations[b]: continue
            dists = []
            for (fy, fx) in fixations[b]:
                dy = yy[b] - fy; dx = xx[b] - fx
                dists.append(torch.sqrt(dy**2 + dx**2))
            d[b] = torch.stack(dists, dim=0).min(0).values
        d_mm = d * self.pixel_mm
        sigma_mm = self.beta * torch.sigmoid(self.alpha * (d_mm - self.mu))
        sigma_px = (sigma_mm / self.pixel_mm).clamp(min=0.0, max=self.max_sigma_px)
        edges = torch.linspace(0, self.max_sigma_px, self.quant_bins+1, device=device)
        bins = torch.bucketize(sigma_px.flatten(), edges) - 1
        bins = bins.clamp(0, self.quant_bins-1).reshape(B,1,H,W)
        out = torch.zeros_like(x)
        for q in range(self.quant_bins):
            mask = (bins == q).float()
            if mask.sum() == 0: continue
            lo, hi = edges[q], edges[q+1]; s = ((lo+hi)/2).item()
            if s < 1e-4:
                out += x * mask; continue
            ksize = int(1 + 2*int(3*s))
            k1d = gaussian_1d(ksize, s, device)
            out += sep_conv(x, k1d) * mask
        return out

def gaussian_1d(ksize, sigma, device):
    ax = torch.arange(ksize, device=device) - (ksize-1)/2
    g = torch.exp(-0.5 * (ax / sigma)**2); g = g / g.sum()
    return g

def sep_conv(x, k1d):
    B, C, H, W = x.shape
    kH = k1d.view(1,1,-1,1); kW = k1d.view(1,1,1,-1)
    xh = F.conv2d(x, weight=kH.repeat(C,1,1,1), groups=C, padding=(k1d.numel()//2,0))
    xw = F.conv2d(xh, weight=kW.repeat(C,1,1,1), groups=C, padding=(0,k1d.numel()//2))
    return xw
