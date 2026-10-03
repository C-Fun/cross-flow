"""Precompute frozen-encoder latents for ImageNet (unflipped + horizontally flipped).

Writes a single memory-mapped array of shape
    (num_images, 2, C, latent_size, latent_size)   [float16]
indexed by the ImageFolder ordering, where [:, 0] is the unflipped latent and
[:, 1] is the horizontally-flipped latent. Training reloads the matching pixel
crop on the fly (see utils/data_util.py), so the crop here MUST match
center_crop_arr.

Encoders (utils/vae_util.py):
  vavae  VA-VAE f16, 32 ch -> 16x16 latents (paper default; official latent stats)
  sdvae  SD-VAE-ft-mse f8, 4 ch -> 32x32 latents (legacy)

Launch (single node, 8 GPUs):
    torchrun --nproc-per-node=8 preprocess_latents.py --encoder vavae \
        --data-dir /path/to/imagenet/train --output /path/to/latents_vavae.npy
"""

import argparse
import os

import numpy as np
from numpy.lib.format import open_memmap
from torchvision.datasets import ImageFolder

import torch

import utils.torch_dist_util as dist
from utils.data_util import center_crop_arr, pixel_to_tensor
from utils.vae_util import build_encoder, ENCODERS


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="ImageNet train directory (class subfolders)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npy memmap path for the latents")
    parser.add_argument("--encoder", type=str, default="vavae", choices=list(ENCODERS.keys()))
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main(args):
    dist.initialize()
    rank = dist.process_index()
    world_size = dist.process_count()

    folder = ImageFolder(args.data_dir)
    samples = folder.samples
    loader = folder.loader
    num_images = len(samples)

    encoder = build_encoder(args.encoder)
    channels = encoder.latent_channels
    latent_size = args.img_size // encoder.downsample

    dist.print0(f"Found {num_images} images; encoder={args.encoder} "
                f"latent {channels}x{latent_size}x{latent_size}")

    # rank 0 creates the memmap file, others wait then open it for writing.
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        open_memmap(
            args.output,
            mode="w+",
            dtype=np.float16,
            shape=(num_images, 2, channels, latent_size, latent_size),
        )
    dist.barrier()
    latents = open_memmap(args.output, mode="r+")

    my_indices = list(range(rank, num_images, world_size))
    device = dist.local_device()

    for start in range(0, len(my_indices), args.batch_size):
        chunk = my_indices[start: start + args.batch_size]

        arrs = [center_crop_arr(loader(samples[i][0]), args.img_size) for i in chunk]
        x_unflip = torch.stack([pixel_to_tensor(a) for a in arrs]).to(device)
        x_flip = torch.stack([pixel_to_tensor(a[:, ::-1]) for a in arrs]).to(device)

        z_unflip = encoder.encode(x_unflip).cpu().half().numpy()
        z_flip = encoder.encode(x_flip).cpu().half().numpy()

        for j, i in enumerate(chunk):
            latents[i, 0] = z_unflip[j]
            latents[i, 1] = z_flip[j]

        if (start // args.batch_size) % 20 == 0:
            done = start + len(chunk)
            print(f"[rank {rank}] {done}/{len(my_indices)}")

    latents.flush()
    dist.barrier()
    dist.print0("Latent precomputation done.")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
