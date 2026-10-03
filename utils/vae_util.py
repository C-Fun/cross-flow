import torch
from diffusers.models import AutoencoderKL as SDAutoencoderKL
from huggingface_hub import hf_hub_download

import utils.torch_util as tu
from utils.vavae_autoencoder import AutoencoderKL as VAVAEAutoencoderKL


# Per-channel SD-VAE latent normalization stats (ImageNet, sd-vae-ft-mse).
# Same convention as the sibling repos: z_norm = (raw_latent - mean) / std.
LATENT_MEAN = [0.86488, -0.27787343, 0.21616915, 0.3738409]
LATENT_STD = [4.85503674, 5.31922414, 3.93725398, 3.9870003]

VAVAE_REPO = "hustvl/vavae-imagenet256-f16d32-dinov2"
VAVAE_CKPT = "vavae-imagenet256-f16d32-dinov2.pt"
VAVAE_STATS = "latents_stats.pt"


class VAEEncoder:
    """Frozen SD-VAE *encoder* (f8, 4 channels), used only for offline latent precomputation.

    CrossFlow needs the encoder E to build clean latents z = E(x0) during
    training; the decoder is never used (the network outputs pixels directly),
    so it is deleted here to save memory.
    """

    latent_channels = 4
    downsample = 8

    def __init__(self, vae_type="mse", dtype=torch.float32):
        vae = SDAutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{vae_type}",
            torch_dtype=dtype,
            local_files_only=False,
        )
        self.dtype = dtype

        # remove decoder (save memory, speed) -- we only ever encode
        del vae.decoder

        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()

        self.vae = tu.device_put(vae)
        self.vae.to(memory_format=torch.channels_last)

        self.mean = torch.tensor(LATENT_MEAN, device=self.vae.device).view(1, -1, 1, 1)
        self.std = torch.tensor(LATENT_STD, device=self.vae.device).view(1, -1, 1, 1)

    @torch.no_grad()
    def encode(self, x):
        """Encode pixel images in [-1, 1] to normalized latent means, (B, 4, H/8, W/8)."""
        x = x.to(self.vae.device, self.dtype).contiguous(
            memory_format=torch.channels_last
        )
        posterior = self.vae.encode(x).latent_dist
        z = (posterior.mean - self.mean) / self.std
        return z


class VAVAEEncoder:
    """Frozen VA-VAE encoder (f16, 32 channels; CrossFlow paper default, Sec 3.3 / App B.2).

    Weights and the official per-channel latent statistics come from the
    hustvl/vavae-imagenet256-f16d32-dinov2 HF repo. Latents are normalized as
    z_norm = (posterior_mean - mean) / std, giving ~unit variance per channel.
    """

    latent_channels = 32
    downsample = 16

    def __init__(self, dtype=torch.float32):
        ckpt = hf_hub_download(VAVAE_REPO, VAVAE_CKPT)
        stats = torch.load(hf_hub_download(VAVAE_REPO, VAVAE_STATS), map_location="cpu")
        vae = VAVAEAutoencoderKL(embed_dim=32, ch_mult=(1, 1, 2, 2, 4), ckpt_path=ckpt)
        self.dtype = dtype

        del vae.decoder
        del vae.post_quant_conv
        for p in vae.parameters():
            p.requires_grad = False
        vae.eval()

        self.vae = tu.device_put(vae).to(dtype)
        self.mean = stats["mean"].to(self.vae.quant_conv.weight.device, dtype)
        self.std = stats["std"].to(self.vae.quant_conv.weight.device, dtype)

    @torch.no_grad()
    def encode(self, x):
        """Encode pixel images in [-1, 1] to normalized latent means, (B, 32, H/16, W/16)."""
        x = x.to(self.vae.quant_conv.weight.device, self.dtype)
        posterior = self.vae.encode(x)
        z = (posterior.mean - self.mean) / self.std
        return z


ENCODERS = {"sdvae": VAEEncoder, "vavae": VAVAEEncoder}


def build_encoder(name, dtype=torch.float32):
    return ENCODERS[name](dtype=dtype)
