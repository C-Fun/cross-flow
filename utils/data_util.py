import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image, image_size):
    """Deterministic center crop (from the DiT/ADM pipeline).

    Resize so the shorter side is `image_size`, then center-crop a square.
    Must be identical between latent precomputation and training so that the
    pixel target x0 stays consistent with the stored latent z.

    Returns:
        HWC uint8 numpy array of shape (image_size, image_size, 3).
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image.convert("RGB"))
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def pixel_to_tensor(arr):
    """HWC uint8 in [0, 255] -> CHW float32 in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 127.5 - 1.0
    return x.permute(2, 0, 1)


class LatentImageNetDataset(Dataset):
    """ImageNet with precomputed (unflipped, flipped) SD-VAE latents.

    Item i returns the pixel target x0 (recomputed with the same deterministic
    center crop used at precompute time) and its matching normalized latent z.
    A random horizontal flip picks between the two stored latents and flips the
    pixel target accordingly, keeping (x0, z) consistent.
    """

    def __init__(self, data_dir, latents_path, img_size=256, flip=True):
        self.folder = ImageFolder(data_dir)
        self.samples = self.folder.samples
        self.loader = self.folder.loader
        self.img_size = img_size
        self.flip = flip

        # (N, 2, C, H, W) float16 memmap: index 0 = unflipped, 1 = flipped.
        self.latents = np.load(latents_path, mmap_mode="r")
        assert self.latents.shape[0] == len(self.samples), (
            f"latents ({self.latents.shape[0]}) and images ({len(self.samples)}) "
            f"count mismatch; regenerate latents for this data_dir"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        arr = center_crop_arr(self.loader(path), self.img_size)

        do_flip = self.flip and (random.random() < 0.5)
        latent_idx = 1 if do_flip else 0
        if do_flip:
            arr = arr[:, ::-1]

        x0 = pixel_to_tensor(arr)
        z = torch.from_numpy(self.latents[i, latent_idx].astype(np.float32))
        return x0, z, label
