import torch
import torch.nn.functional as F

@torch.no_grad()
def lesion_mask_from_cams(cam_map, thr=0.5):
    roi = (cam_map >= thr).float()
    M = 1.0 - roi
    return M

def upsample_cam(cam_small, H, W):
    return F.interpolate(cam_small, size=(H,W), mode='bilinear', align_corners=False)
