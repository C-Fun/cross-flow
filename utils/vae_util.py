import torch
from diffusers.models import AutoencoderKL
import utils.torch_util as tu


# Per-channel SD-VAE latent normalization stats (ImageNet, sd-vae-ft-mse).
# Same convention as the sibling repos: z_norm = (raw_latent - mean) / std.
LATENT_MEAN = [0.86488, -0.27787343, 0.21616915, 0.3738409]
LATENT_STD = [4.85503674, 5.31922414, 3.93725398, 3.9870003]


class VAEEncoder:
    """Frozen SD-VAE *encoder*, used only for offline latent precomputation.

    CrossFlow needs the encoder E to build clean latents z = E(x0) during
    training; the decoder is never used (the network outputs pixels directly),
    so it is deleted here to save memory.
    """

    def __init__(self, vae_type="mse", dtype=torch.float32):
        vae = AutoencoderKL.from_pretrained(
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
        """Encode pixel images in [-1, 1] to normalized latent means.

        Args:
            x: (B, 3, H, W) float tensor in [-1, 1].

        Returns:
            z: (B, 4, H/8, W/8) normalized latent (posterior mean).
        """
        x = x.to(self.vae.device, self.dtype).contiguous(
            memory_format=torch.channels_last
        )
        posterior = self.vae.encode(x).latent_dist
        z = (posterior.mean - self.mean) / self.std
        return z
