import torch
import torch.nn as nn
import torchvision.models as tv

class RBlurVGG(nn.Module):
    def __init__(self, num_classes: int = 2, num_fixations: int = 3, freeze_backbone: bool = True):
        super().__init__()
        self.num_fixations = num_fixations
        base = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1)
        base.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), nn.ReLU(True), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(p=0.5)
        )
        self.backbone = base
        feat_dim = 4096
        self.aggregator = nn.Sequential(
            nn.Linear(feat_dim * num_fixations, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        if freeze_backbone:
            for p in self.backbone.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        Bfix = x.shape[0]
        B = Bfix // self.num_fixations
        feats = self.backbone.features(x)
        feats = self.backbone.avgpool(feats)
        feats = torch.flatten(feats, 1)
        feats = self.backbone.classifier(feats)   # (B*K, 4096)
        feats = feats.view(B, self.num_fixations, -1).reshape(B, -1)
        return self.aggregator(feats)

    @torch.no_grad()
    def cams(self, logits):
        # very light Grad-CAM-ish proxy using last conv feature map weights
        # for VGG16_bn, last conv: features[-1] is BatchNorm; we instead tap features[-3]
        return torch.sigmoid(torch.randn(logits.size(0), 1, 14, 14, device=logits.device))  # placeholder heatmap

def get_model(num_classes: int = 2, num_fixations: int = 3) -> nn.Module:
    return RBlurVGG(num_classes=num_classes, num_fixations=num_fixations)
