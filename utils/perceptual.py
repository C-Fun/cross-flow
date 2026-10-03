"""Pixel-space perceptual losses for CrossFlow.

The paper (App. B.4) maps images to DINOv3-B features and takes a Huber loss
between the feature tensors of the reconstruction-compatible prediction and the
target. DINOv3-B weights are available ungated via timm; DINOv2-B is kept as a
fallback and VGG-LPIPS as an ablation option.
"""

import timm
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

TIMM_NAMES = {
    "dinov3": "vit_base_patch16_dinov3.lvd1689m",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoPerceptual(nn.Module):
    """Huber distance between frozen DINO ViT-B feature tensors (all tokens).

    Inputs are images in [-1, 1]. 256x256 inputs are fed natively (DINOv3 patch
    16 -> the same 16x16 token grid as the generator); tiny images (CIFAR) are
    bilinearly upsampled to 224 so the backbone sees a sensible scale.
    """

    def __init__(self, net="dinov3", min_res=224):
        super().__init__()
        self.backbone = timm.create_model(
            TIMM_NAMES[net], pretrained=True, num_classes=0, dynamic_img_size=True
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.min_res = min_res
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):  # keep the frozen backbone in eval mode
        return super().train(False)

    def features(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        if x.shape[-1] < self.min_res:
            x = F.interpolate(x, size=(self.min_res, self.min_res), mode="bilinear", align_corners=False)
        return self.backbone.forward_features(x)

    def forward(self, pred, target):
        fp = self.features(pred)
        with torch.no_grad():
            ft = self.features(target)
        return F.huber_loss(fp.float(), ft.float())


class LPIPSPerceptual(nn.Module):
    """VGG-LPIPS wrapper with the same (pred, target) -> scalar interface."""

    def __init__(self, net="vgg"):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        return super().train(False)

    def forward(self, pred, target):
        return self.lpips(pred.float(), target.float()).mean()


def build_perceptual(name):
    if name in TIMM_NAMES:
        return DinoPerceptual(net=name)
    if name == "lpips":
        return LPIPSPerceptual()
    raise ValueError(f"unknown perceptual net: {name}")
