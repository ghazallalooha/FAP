import os, random, math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.v2 import functional as F_v2
from PIL import Image

class FastGaussianBlur:
    def __call__(self, tensor, kernel_size, sigma):
        x = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, device=tensor.device)
        x = torch.exp(-x**2/(2*sigma**2)); x = x / x.sum()
        kernel = x[:, None] * x[None, :]
        kernel = kernel.expand(tensor.size(1), 1, kernel_size, kernel_size)
        pad = kernel_size // 2
        return F.conv2d(tensor, kernel, padding=pad, groups=tensor.size(1))

class RBlurTransform:
    def __init__(self, device='cpu', Wv=224, sigma_c=0.12, sigma_r=0.09, noise_scale=0.0):
        self.device = torch.device(device)
        self.Wv, self.sigma_c, self.sigma_r, self.noise_scale = Wv, sigma_c, sigma_r, noise_scale
        self.gauss = FastGaussianBlur()
        self.base = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    def __call__(self, img, fixation_xy):
        img = self.base(img)
        t = F_v2.to_image(img).to(self.device)
        t = F_v2.to_dtype(t, torch.float32, scale=True)
        fx, fy = fixation_xy
        r = math.sqrt((fx-0.5)**2 + (fy-0.5)**2)
        sigma = self.sigma_c + self.sigma_r * r
        ksize = int(2 * math.ceil(3 * sigma) + 1)
        out = self.gauss(t.unsqueeze(0), ksize, sigma)
        return F_v2.normalize(out.squeeze(0),
                              mean=[0.485,0.456,0.406],
                              std=[0.229,0.224,0.225])

class CMMDDataset(Dataset):
    def __init__(self, root, split='train', num_fixations=3, device='cpu'):
        self.root = root
        self.split = split
        self.num_fixations = max(1, int(num_fixations))
        self.df = pd.read_csv(os.path.join(root, split, f'{split}_labels.csv'))
        self.tx = RBlurTransform(device=device)

    def _rand_fixations(self):
        return [(random.random(), random.random()) for _ in range(self.num_fixations)]

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        p = self.df.iloc[i]['image_path']
        y = int(self.df.iloc[i]['label'])
        img = Image.open(p).convert('RGB')
        stack = [self.tx(img, fix) for fix in self._rand_fixations()]
        return torch.stack(stack, dim=0), torch.tensor(y, dtype=torch.long)

def get_cmmd_dataloader(root, split='train', batch_size=16, num_fixations=3, device='cpu', shuffle=None):
    ds = CMMDDataset(root, split, num_fixations=num_fixations, device=device)
    if shuffle is None:
        shuffle = (split == 'train')
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
