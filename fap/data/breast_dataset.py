import os, random, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms.v2 import functional as F_v2
from PIL import Image
import torch.nn.functional as F

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

    def __call__(self, img):
        img = self.base(img)
        t = F_v2.to_image(img).to(self.device)
        t = F_v2.to_dtype(t, torch.float32, scale=True)
        r = 0.0
        sigma = self.sigma_c + self.sigma_r * r
        ksize = int(2 * math.ceil(3 * sigma) + 1)
        out = self.gauss(t.unsqueeze(0), ksize, sigma)
        return F_v2.normalize(out.squeeze(0),
                              mean=[0.485,0.456,0.406],
                              std=[0.229,0.224,0.225])

class BreastDataset(Dataset):
    def __init__(self, data_dir, csv_path, num_fixations=3, device='cpu'):
        self.data_dir = data_dir
        self.meta = pd.read_csv(csv_path, dtype={'PatientID': str})
        self.lab = dict(zip(self.meta['PatientID'].astype(str), self.meta['Cancer'].astype(int)))
        self.paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    pid = os.path.splitext(f)[0]
                    if pid in self.lab:
                        self.paths.append(os.path.join(root, f))
        self.labels = [self.lab[os.path.splitext(os.path.basename(p))[0]] for p in self.paths]
        self.num_fixations = num_fixations
        self.tx = RBlurTransform(device=device)

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        stack = [self.tx(img) for _ in range(self.num_fixations)]
        return torch.stack(stack, 0), torch.tensor(self.labels[i], dtype=torch.long)

def get_breast_loader(data_dir, csv_path, batch_size=32, num_fixations=3, device='cpu'):
    ds = BreastDataset(data_dir, csv_path, num_fixations=num_fixations, device=device)
    labels = np.array(ds.labels)
    uniq = np.unique(labels)
    class_counts = np.array([np.sum(labels==c) for c in uniq])
    class_weights = (1.0 / class_counts)
    weights = class_weights[labels]
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float), len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
