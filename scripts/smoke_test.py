"""Fast single-GPU sanity check for the CrossFlow pipeline (no data needed).

Validates: model construction, the cross-space JVP loss, backprop to params,
and one-step generation. Run on an MI210 node before launching real training:

    python scripts/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import utils.torch_util as tu
from crossflow import CrossFlow


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda", 0)
    torch.manual_seed(0)

    model_str = "crossflowDiT_B_2"
    model = CrossFlow(model_str, latent_size=32, num_classes=1000).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_str}: {n_params/1e6:.1f}M params, pixel_size={model.img_size}")

    bsz = 4
    z = torch.randn(bsz, 4, 32, 32, device=device)
    x0 = torch.randn(bsz, 3, model.img_size, model.img_size, device=device).clamp(-1, 1)
    y = torch.randint(0, 1000, (bsz,), device=device)

    # forward + cross-space JVP loss
    loss_cf, x_pred, diagonal = model.compute_loss(z, x0, y)
    print(f"loss_cf={loss_cf.item():.4f}  x_pred={tuple(x_pred.shape)}  "
          f"diagonal={diagonal.sum().item()}/{bsz}")
    assert x_pred.shape == x0.shape

    # backprop reaches the network params
    loss_cf.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients -- JVP primal did not connect to params!"
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    assert torch.isfinite(total_norm), "non-finite gradient"
    print(f"grad ok: {len(grads)} tensors, total_norm={total_norm.item():.3f}")

    # one-step generation
    model.eval()
    imgs = model.generate(
        n_sample=bsz,
        rng=tu.BatchGenerator(device=device, seeds=torch.arange(bsz)),
        y=y,
        cfg_omega=1.0,
    )
    print(f"generate ok: {tuple(imgs.shape)}")

    # cfg path
    imgs_cfg = model.generate(
        n_sample=bsz,
        rng=tu.BatchGenerator(device=device, seeds=torch.arange(bsz)),
        y=y,
        cfg_omega=2.0,
    )
    print(f"generate (cfg) ok: {tuple(imgs_cfg.shape)}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
