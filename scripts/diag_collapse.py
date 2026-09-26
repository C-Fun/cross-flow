"""Diagnose why CrossFlow samples carry no semantics (run on one GPU, no training).

Checks, using the EMA weights of a checkpoint:
  A. class sensitivity of F(eps, r=0, y): does changing y change the output at all?
  B. diagonal reconstruction F(z_t, r=t) vs x0 across t: decoder-collapse signature
     is "great at small t, garbage at large t".
  C. JVP correctness: torch.func.jvp tangent vs central finite difference (fp32),
     and bf16-autocast jvp vs fp32 jvp.
  D. gradient budget: pre-clip grad norm from t-bucketed subsets, to see which
     time regime dominates the (clipped) update.

    python scripts/diag_collapse.py --ckpt runs/crossflow_B_2/latest.pt \
        --data-dir $WORK/dataset/imagenet/train \
        --latents-path $WORK/dataset/imagenet-latents/train/latents.npy --out diag
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
from torch.func import jvp

import utils.torch_util as tu
from utils.data_util import LatentImageNetDataset
from crossflow import CrossFlow, broadcast_time

torch.backends.cudnn.benchmark = False


def to_uint8(x):
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) / 2 * 255).round().byte()
    return x.permute(0, 2, 3, 1).numpy()


def psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(4.0 / mse)).tolist(), mse.tolist()


def residual_loss(model, z, x0, y, t, r, noise, autocast=None):
    """Same math as CrossFlow.compute_loss but with explicit (t, r, noise)."""
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if autocast else torch.autocast("cuda", enabled=False)
    with ctx:
        z_t = (1 - broadcast_time(t, z)) * z + broadcast_time(t, z) * noise
        v = noise - z
        x_pred, dx_dt = jvp(lambda zi: model.net(zi, r, y), (z_t,), (v,))
        dx_dt = dx_dt.detach()
        t_x, r_x = broadcast_time(t, x_pred), broadcast_time(r, x_pred)
        res = (r_x / t_x.square()) * (x_pred - x0) + (1 - r_x / t_x) * dx_dt
    return res.float().square().mean(), x_pred, dx_dt


def grad_norm_of(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
    model.zero_grad(set_to_none=True)
    return g.item()


def main(a):
    dev = torch.device("cuda", 0)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)

    model = CrossFlow(a.model, latent_size=32, num_classes=1000, time_eps=a.time_eps).to(dev)
    ck = torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ck["ema" if not a.raw else "model"])
    print(f"loaded step={ck.get('step')} weights={'model' if a.raw else 'ema'}")
    model.eval()

    ds = LatentImageNetDataset(a.data_dir, a.latents_path, img_size=256, flip=False)
    idx = np.linspace(0, len(ds) - 1, 8).astype(int)
    batch = [ds[int(i)] for i in idx]
    x0 = torch.stack([b[0] for b in batch]).to(dev)
    z = torch.stack([b[1] for b in batch]).to(dev)
    y = torch.tensor([b[2] for b in batch], device=dev)
    print(f"data: x0 {tuple(x0.shape)} z {tuple(z.shape)} z.std={z.std():.3f} labels={y.tolist()}")

    # ---------------- A. class sensitivity ----------------
    print("\n[A] class sensitivity of one-step generation F(eps, r=0, y)")
    with torch.no_grad():
        n = 8
        eps = torch.randn(n, 4, 32, 32, device=dev)
        r0 = torch.zeros(n, device=dev)
        ys = [torch.full((n,), c, device=dev) for c in (0, 207, 500, 980)]
        outs = [model.net(eps, r0, yy) for yy in ys]
        cls_diff = torch.stack([(outs[i] - outs[j]).abs().mean() for i in range(4) for j in range(i + 1, 4)]).mean()
        eps2 = torch.randn_like(eps)
        noise_diff = (model.net(eps2, r0, ys[0]) - outs[0]).abs().mean()
        y_null = torch.full((n,), 1000, device=dev)
        null_diff = (model.net(eps, r0, y_null) - outs[0]).abs().mean()
        print(f"  mean|dF| changing CLASS (fixed eps): {cls_diff:.4f}")
        print(f"  mean|dF| changing NOISE (fixed y) : {noise_diff:.4f}")
        print(f"  mean|dF| cond vs NULL label       : {null_diff:.4f}")
        print(f"  -> class/noise ratio = {cls_diff / noise_diff:.3f}  (<<1 means class is ignored)")
        gen = torch.cat([o[:4] for o in outs], 0)
        cv2.imwrite(os.path.join(a.out, "A_class_rows_same_eps.png"),
                    to_uint8(gen).reshape(4, 4, 256, 256, 3).transpose(0, 2, 1, 3, 4).reshape(1024, 1024, 3)[:, :, ::-1])

    # ---------------- B. reconstruction across t ----------------
    print("\n[B] diagonal reconstruction F(z_t, r=t) and endpoint F(z_t, r~0) vs x0")
    ts = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]
    rows = [x0]
    with torch.no_grad():
        noise = torch.randn_like(z)
        for tv in ts:
            t = torch.full((8,), tv, device=dev)
            z_t = (1 - tv) * z + tv * noise
            rec = model.net(z_t, t, y)
            end = model.net(z_t, torch.full((8,), a.time_eps, device=dev), y)
            p_rec, _ = psnr(rec, x0)
            p_end, _ = psnr(end, x0)
            print(f"  t={tv:4.2f} | diag r=t  PSNR {np.mean(p_rec):5.2f} dB | endpoint r~0 PSNR {np.mean(p_end):5.2f} dB "
                  f"| |rec-end| {(rec - end).abs().mean():.4f}")
            rows.append(rec)
        grid = torch.stack(rows, 0)[:, :4]  # (rows, 4, C,H,W)
        img = to_uint8(grid.reshape(-1, 3, 256, 256)).reshape(len(rows), 4, 256, 256, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(len(rows) * 256, 4 * 256, 3)
        cv2.imwrite(os.path.join(a.out, "B_recon_rows_x0_then_t.png"), img[:, :, ::-1])

    # ---------------- C. JVP correctness ----------------
    print("\n[C] JVP check at t=0.5 (2 samples)")
    zs, x0s, ys2 = z[:2], x0[:2], y[:2]
    noise = torch.randn_like(zs)
    t = torch.full((2,), 0.5, device=dev)
    r = torch.full((2,), 0.25, device=dev)
    z_t = 0.5 * zs + 0.5 * noise
    v = noise - zs
    with torch.no_grad():
        f = lambda zi: model.net(zi, r, ys2)
        _, j32 = jvp(f, (z_t,), (v,))
        h = 1e-3
        fd = (f(z_t + h * v) - f(z_t - h * v)) / (2 * h)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, j16 = jvp(f, (z_t,), (v,))
        j16 = j16.float()
        rel = lambda p, q: ((p - q).norm() / (q.norm() + 1e-12)).item()
        print(f"  |jvp fp32|={j32.norm():.3f} |finite-diff|={fd.norm():.3f} |jvp bf16|={j16.norm():.3f}")
        print(f"  rel err jvp_fp32 vs finite-diff : {rel(j32, fd):.4f}   (should be ~1e-3..1e-2)")
        print(f"  rel err jvp_bf16 vs jvp_fp32    : {rel(j16, j32):.4f}   (>0.3 means bf16 JVP is unreliable)")
        cos = torch.nn.functional.cosine_similarity(j16.flatten(), j32.flatten(), dim=0).item()
        print(f"  cosine(jvp_bf16, jvp_fp32)      : {cos:.4f}")
        print(f"  |dF/dt| per-pixel rms = {j32.pow(2).mean().sqrt():.4f}  vs |F-x0| rms = {(f(z_t) - x0s).pow(2).mean().sqrt():.4f}")

    # ---------------- D. gradient budget by time regime ----------------
    print("\n[D] pre-clip grad norm by time regime (fp32, batch=8, EMA weights, same noise)")
    model.train()
    noise = torch.randn_like(z)
    for name, tv, rv in [
        ("diag small t   (t=r=0.01)", 0.01, 0.01),
        ("diag tiny t    (t=r=1e-3)", 1e-3, 1e-3),
        ("diag mid t     (t=r=0.5)", 0.5, 0.5),
        ("diag large t   (t=r=0.95)", 0.95, 0.95),
        ("gen regime     (t=0.95,r=0.05)", 0.95, 0.05),
        ("gen regime     (t=0.95,r=1e-3)", 0.95, 1e-3),
        ("mid off-diag   (t=0.6,r=0.3)", 0.6, 0.3),
    ]:
        t = torch.full((8,), tv, device=dev)
        r = torch.full((8,), rv, device=dev)
        loss, _, dxdt = residual_loss(model, z, x0, y, t, r, noise)
        g = grad_norm_of(model, loss)
        w = rv / tv ** 2
        print(f"  {name:32s} weight r/t^2={w:9.1f}  loss={loss.item():12.2f}  gradnorm={g:12.2f}  |dF/dt|rms={dxdt.pow(2).mean().sqrt():.3f}")

    print("\n[D2] fraction of loss/grad from t<0.05 under the actual sampler (1 batch of 64 draws)")
    tt, rr, dg = model.sample_times(4096, dev)
    small = (tt < 0.05).float().mean().item()
    print(f"  P(t<0.05)={small:.3f}  P(diag)={dg.float().mean():.3f}  mean weight r/t^2={(rr / tt**2).mean():.1f}  "
          f"max={(rr / tt**2).max():.1f}  median={(rr / tt**2).median():.2f}")
    print("done ->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--latents-path", required=True)
    p.add_argument("--out", default="diag")
    p.add_argument("--model", default="crossflowDiT_B_2")
    p.add_argument("--time-eps", type=float, default=1e-4)
    p.add_argument("--raw", action="store_true", help="use raw model weights instead of EMA")
    main(p.parse_args())
