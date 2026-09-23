import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedder import BottleneckPatchEmbedder, TimestepEmbedder, LabelEmbedder
from models.torch_models import TorchLinear, RMSNorm, SwiGLUMlp


def unsqueeze(t, dim):
    """Adds a new axis to a tensor at the given position."""
    return t.unsqueeze(dim)


#################################################################################
#                   Modern Transformer Components with Vec Gates                #
#################################################################################


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, rope_angles):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_angles)
        k = apply_rotary_pos_emb(k, rope_angles)

        # manually implement attention to match JAX implementation
        query = q / math.sqrt(self.head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)

        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=8 / 3,
        weight_init="scaled_variance",
        weight_init_constant=1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.weight_init = weight_init
        self.weight_init_constant = weight_init_constant

        self.norm1 = RMSNorm(self.hidden_size)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        # round mlp hidden dim to multiple of 8
        if hidden_size > 1024:  # only for HSDP code
            mlp_hidden_dim = (mlp_hidden_dim + 7) // 8 * 8
        self.mlp = SwiGLUMlp(
            self.hidden_size,
            mlp_hidden_dim,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )

        self.attn_scale = nn.Parameter(torch.zeros(self.hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(self.hidden_size))

    def forward(self, x, rope_angles):
        x = x + self.attn(self.norm1(x), rope_angles) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    def __init__(self, hidden_size, out_patch_size, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_patch_size = out_patch_size
        self.out_channels = out_channels

        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.out_patch_size * self.out_patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def forward(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                CrossFlow DiT with In-context Conditioning                     #
#################################################################################


class crossflowDiT(nn.Module):
    """
    CrossFlow cross-space Transformer (crossflowDiT).

    The input is a noised *latent* z_t (latent_channels x latent_size x latent_size)
    and the output F_theta is a *pixel* image (out_channels x pixel_size x pixel_size).
    A latent patch-embed maps z_t to a (grid x grid) token sequence; a pixel
    unpatchify head maps the same token grid back to full-resolution pixels.
    Conditioning is in-context via learnable prefix tokens (class label + target
    time r). The network does not condition on t explicitly, only on r
    (paper appendix B.2).
    """

    def __init__(
        self,
        latent_size: int = 32,
        patch_size: int = 2,
        latent_channels: int = 4,
        out_patch_size: int = 16,
        out_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 8 / 3,
        num_classes: int = 1000,
        num_class_tokens: int = 8,
        num_time_tokens: int = 4,
        token_init_constant: float = 1.0,
        embedding_init_constant: float = 1.0,
        weight_init_constant: float = 0.32,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.out_patch_size = out_patch_size
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.num_class_tokens = num_class_tokens
        self.num_time_tokens = num_time_tokens

        self.token_init_constant = token_init_constant
        self.embedding_init_constant = embedding_init_constant
        self.weight_init_constant = weight_init_constant

        # grid of tokens (shared between latent input and pixel output)
        assert latent_size % patch_size == 0
        self.grid_size = latent_size // patch_size
        self.pixel_size = self.grid_size * self.out_patch_size

        self.x_embedder = BottleneckPatchEmbedder(
            self.latent_size,
            self.patch_size,
            128 if self.hidden_size <= 1024 else 256,  # pca channels. 256 for H/G
            self.latent_channels,
            self.hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
        )

        self.r_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(self.num_classes, **embed_kwargs)

        token_initializer = partial(
            nn.init.normal_, std=self.token_init_constant / math.sqrt(self.hidden_size)
        )
        self.time_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_time_tokens, self.hidden_size))
        )
        self.class_tokens = nn.Parameter(
            token_initializer(torch.empty(1, self.num_class_tokens, self.hidden_size))
        )

        total_tokens = (
            self.x_embedder.num_patches
            + self.num_class_tokens
            + self.num_time_tokens
        )
        self.prefix_tokens = self.num_class_tokens + self.num_time_tokens
        self.head_dim = self.hidden_size // self.num_heads
        self.register_buffer(
            "rope_angles",
            precompute_rope_freqs(self.head_dim, self.x_embedder.num_patches),
        )
        self.pos_embed = nn.Parameter(
            nn.init.normal_(torch.empty(1, total_tokens, self.hidden_size), std=0.02)
        )

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(**block_kwargs) for _ in range(self.depth)]
        )
        self.final_layer = FinalLayer(
            self.hidden_size, self.out_patch_size, self.out_channels
        )

    def unpatchify(self, x):
        """(B, num_patches, out_patch**2 * C) -> (B, C, pixel_size, pixel_size)."""
        c = self.out_channels
        p = self.out_patch_size
        h = w = self.grid_size
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        images = x.reshape((x.shape[0], c, h * p, w * p))
        return images

    def _build_sequence(self, x, r, y):
        """
        Build the input token sequence for the transformer.
        1. Embed the input latent patches.
        2. Embed the conditioning information (target time r, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.
        """
        x_embed = self.x_embedder(x)
        r_embed = self.r_embedder(r)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + unsqueeze(r_embed, 1)
        class_tokens = self.class_tokens + unsqueeze(y_embed, 1)

        seq = torch.cat([class_tokens, time_tokens, x_embed], axis=1)
        seq = seq + self.pos_embed
        return seq

    def forward(self, x, r, y):
        """
        Forward pass of the crossflowDiT model.

        Args:
            x: Noised latent z_t, shape (B, latent_channels, latent_size, latent_size).
            r: Target time r in [0, 1], shape (B,).
            y: Class labels, shape (B,).

        Returns:
            F_theta: Predicted pixel image, shape (B, out_channels, pixel_size, pixel_size).
        """
        seq = self._build_sequence(x, r, y)

        for block in self.blocks:
            seq = block(seq, self.rope_angles)

        tokens = seq[:, self.prefix_tokens:]
        images = self.unpatchify(self.final_layer(tokens))
        return images


#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    """Precompute 2D-RoPE rotation *angles* (real), shape (seq_len, dim // 2).

    We keep the angles (rather than complex cos/sin) so that the rotation can be
    applied with plain real arithmetic in `apply_rotary_pos_emb`, which composes
    cleanly with forward-mode autodiff (torch.func.jvp) used by CrossFlow.
    """
    dim = dim // 2  # for 2d rotary embeddings
    T = int(seq_len ** 0.5)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(T, dtype=torch.float32)
    freqs_h = torch.einsum("i,j->ij", positions, freqs)
    freqs_w = torch.einsum("i,j->ij", positions, freqs)
    freqs = torch.cat(
        [
            torch.tile(freqs_h[:, None, :], (1, T, 1)),
            torch.tile(freqs_w[None, :, :], (T, 1, 1)),
        ],
        axis=-1,
    )  # (T, T, dim)
    angles = freqs.reshape(seq_len, dim)
    return angles


def apply_rotary_pos_emb(x, angles):
    """Apply 2D RoPE to the last P image-patch tokens; prefix tokens are untouched.

    Args:
        x: (B, S, num_heads, head_dim)
        angles: (P, head_dim // 2), P = number of image patches
    """
    P = angles.shape[0]
    cos = torch.cos(angles).to(x.dtype)[None, :, None, :]  # (1, P, 1, head_dim//2)
    sin = torch.sin(angles).to(x.dtype)[None, :, None, :]

    x_img = x[:, -P:]
    x1 = x_img[..., 0::2]
    x2 = x_img[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    x_rot = torch.stack([rot1, rot2], dim=-1).flatten(-2)

    return torch.cat([x[:, :-P], x_rot], dim=1)


#################################################################################
#                             CrossFlow DiT Configs                             #
#################################################################################


crossflowDiT_B_2 = partial(
    crossflowDiT,
    latent_size=32,
    patch_size=2,
    out_patch_size=16,
    depth=12,
    hidden_size=768,
    num_heads=12,
)

crossflowDiT_L_2 = partial(
    crossflowDiT,
    latent_size=32,
    patch_size=2,
    out_patch_size=16,
    depth=24,
    hidden_size=1024,
    num_heads=16,
)

crossflowDiT_XL_2 = partial(
    crossflowDiT,
    latent_size=32,
    patch_size=2,
    out_patch_size=16,
    depth=28,
    hidden_size=1152,
    num_heads=16,
)
