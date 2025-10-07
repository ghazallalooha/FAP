import torch

@torch.no_grad()
def gradient_guided_fixations(model, x, y, loss_fn, K=3, gamma=0.1):
    """
    Returns a list[batch] of K fixation points (y,x) based on grad saliency.
    """
    model.eval()
    x = x.detach().requires_grad_(True)
    logits = model(x)
    loss = loss_fn(logits, y)
    grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
    sal = grad.norm(p=2, dim=1, keepdim=True)  # (B,1,H,W)
    s = (sal - sal.amin(dim=(-1,-2), keepdim=True))
    s = s / (s.amax(dim=(-1,-2), keepdim=True) + 1e-8)
    prob = torch.sigmoid(gamma * s)

    B, _, H, W = prob.shape
    fixations = [[] for _ in range(B)]
    heat = prob.clone()
    for _ in range(K):
        flat_idx = heat.view(B, -1).argmax(dim=1)
        yy = (flat_idx // W).tolist(); xx = (flat_idx % W).tolist()
        for b in range(B):
            fixations[b].append((yy[b], xx[b]))
            r = max(5, int(0.03*min(H,W)))
            y0, x0 = yy[b], xx[b]
            y1, y2 = max(0,y0-r), min(H, y0+r+1)
            x1, x2 = max(0,x0-r), min(W, x0+r+1)
            heat[b,0,y1:y2,x1:x2] *= 0.1
    return fixations
