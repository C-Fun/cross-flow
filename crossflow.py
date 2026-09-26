import torch
import torch.nn as nn
from torch.func import jvp

from models import crossflowDiT


def broadcast_time(s, x):
    """Reshape a per-sample [batch] coefficient to broadcast against x."""
    return s.reshape(s.shape[0], *([1] * (x.ndim - 1)))


class CrossFlow(nn.Module):
    """CrossFlow: one-step generation across latent and pixel spaces.

    The network takes a noised latent z_t (latent space) and a target time r,
    and directly outputs a pixel image F_theta. Training uses the differential
    cross-space objective (paper eq. 11'); one-step inference is x_hat = F(eps, 0, y).
    """

    def __init__(
        self,
        model_str: str,
        dtype: torch.dtype = torch.float32,
        latent_size: int = 32,
        latent_channels: int = 4,
        img_channels: int = 3,
        num_classes: int = 1000,
        time_eps: float = 1e-2,
        diagonal_prob: float = 0.5,
        adaptive_p: float = 1.0,
    ):
        super().__init__()
        self.model_str = model_str
        self.dtype = dtype
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.time_eps = time_eps
        self.diagonal_prob = diagonal_prob
        self.adaptive_p = adaptive_p

        net_fn = getattr(crossflowDiT, self.model_str)
        self.net: crossflowDiT.crossflowDiT = net_fn(
            latent_size=self.latent_size,
            latent_channels=self.latent_channels,
            out_channels=self.img_channels,
            num_classes=self.num_classes,
        )
        self.img_size = self.net.pixel_size

    #######################################################
    #                    Time sampling                    #
    #######################################################

    def sample_times(self, batch_size, device):
        """Shifted-time sampling with a `diagonal_prob` chance of r == t.

        Follows the CrossFlow pseudo-code: draw two shifted-time samples per
        example, sort them into (t >= r); with probability `diagonal_prob`
        collapse them to a common time so the sample lands on the r == t
        reconstruction anchor.
        """
        latent_dim = self.latent_channels * self.latent_size * self.latent_size
        shift = (latent_dim / 4096) ** 0.5

        u = torch.rand(batch_size, 2, device=device)
        times = shift * u / (1 + (shift - 1) * u)
        times = times.clamp(self.time_eps, 1 - self.time_eps)

        t = times.max(dim=1).values
        r = times.min(dim=1).values

        diagonal = torch.rand(batch_size, device=device) < self.diagonal_prob
        t = torch.where(diagonal, times[:, 0], t)
        r = torch.where(diagonal, times[:, 0], r)
        return t, r, diagonal

    #######################################################
    #              Cross-space training loss              #
    #######################################################

    def compute_loss(self, z, x0, y):
        """Compute the CrossFlow residual loss (paper eq. 11').

        Args:
            z: Clean latents E(x0), normalized, shape (B, Cz, Hz, Wz).
            x0: Target pixel images in [-1, 1], shape (B, C, H, W).
            y: Class labels (possibly with CFG null label), shape (B,).

        Returns:
            loss: scalar loss to backprop (adaptively normalized if adaptive_p > 0).
            x_pred: pixel prediction F_theta(z_t, r, y), shape like x0 (keeps grad).
            diagonal: bool mask of r == t samples, shape (B,).
            loss_cf: raw (un-normalized) mean squared residual, for logging.
        """
        device = z.device
        batch_size = z.shape[0]

        noise = torch.randn_like(z)
        t, r, diagonal = self.sample_times(batch_size, device)

        # eq. (1): linear noising path in latent space.
        t_z = broadcast_time(t, z)
        z_t = (1 - t_z) * z + t_z * noise

        # eq. (3): conditional velocity direction for the JVP tangent.
        velocity = noise - z

        # F conditions only on r (not t); dt-derivative flows through z_t only,
        # so the JVP tangent is the conditional velocity on the latent input.
        def net_fn(z_input):
            return self.net(z_input, r, y)

        x_pred, dx_dt = jvp(net_fn, (z_t,), (velocity,))

        # paper stop-grads the JVP term.
        dx_dt = dx_dt.detach()

        # eq. (11'): avoids the explicit 1/r division; on the r == t diagonal
        # this reduces to the reconstruction residual (1/t)(F_theta - x0).
        t_x = broadcast_time(t, x_pred)
        r_x = broadcast_time(r, x_pred)
        residual = (r_x / t_x.square()) * (x_pred - x0) + (1 - r_x / t_x) * dx_dt

        # per-sample squared residual (mean over pixels)
        sq = residual.square().flatten(1).mean(1)
        loss_cf = sq.mean()

        # Adaptive per-sample normalization (MeanFlow-style): w = 1 / sg(||R||^2 + c)^p.
        # Keeps the paper's residual/target direction but equalizes gradient magnitude
        # across (t, r). Without it the r/t^2 factor lets tiny-t reconstruction samples
        # produce gradients 50-5000x larger than the rest, and gradient clipping then
        # turns every update into "decode a nearly-clean latent" (measured collapse).
        if self.adaptive_p > 0:
            w = 1.0 / (sq.detach() + 1e-3).pow(self.adaptive_p)
            loss = (w * sq).mean()
        else:
            loss = loss_cf
        return loss, x_pred, diagonal, loss_cf

    #######################################################
    #                   One-step sampling                 #
    #######################################################

    @torch.no_grad()
    def generate(self, n_sample, rng, y=None, cfg_omega=1.0):
        """One-step generation: x_hat = F(eps, r=0, y).

        Args:
            n_sample: number of images to generate.
            rng: a BatchGenerator providing per-sample deterministic noise.
            y: optional class labels; random if None.
            cfg_omega: classifier-free guidance scale on the pixel output
                (1.0 disables guidance / single forward).
        """
        z_shape = (n_sample, self.latent_channels, self.latent_size, self.latent_size)
        eps = rng.randn(z_shape).to(self.dtype)
        r = torch.zeros(n_sample, dtype=self.dtype, device=eps.device)

        if y is not None:
            y = y.to(eps.device).long()
        else:
            y = rng.randint(
                0, self.num_classes, size=(n_sample,), dtype=torch.int64
            ).to(eps.device)

        if cfg_omega == 1.0:
            x = self.net(eps, r, y)
        else:
            y_null = torch.full_like(y, self.num_classes)
            x_cond = self.net(eps, r, y)
            x_uncond = self.net(eps, r, y_null)
            x = x_uncond + cfg_omega * (x_cond - x_uncond)

        return x
