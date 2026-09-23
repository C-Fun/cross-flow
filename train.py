"""CrossFlow training on ImageNet (class-conditional, cross-space latent->pixel).

Single-node multi-GPU via torchrun. We do NOT wrap the model in
DistributedDataParallel because torch.func.jvp (used for dF/dt) does not compose
with DDP; instead we broadcast parameters once and manually all-reduce gradients.

Checkpoints are written to {workdir}/latest.pt (atomically) plus periodic
{workdir}/ckpt_{step}.pt, and training auto-resumes from latest.pt on startup.
When --total-steps is reached a {workdir}/DONE marker is written so the
self-resubmitting Slurm chain knows to stop.
"""

import argparse
import hashlib
import os
import time
from contextlib import nullcontext

import numpy as np
import cv2
import lpips
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.torch_dist_util as dist
import utils.torch_util as tu
from utils.data_util import LatentImageNetDataset
from utils.ema_util import EMA
from crossflow import CrossFlow
from evaluate import run_evaluate

# evaluate.py turns on cudnn.benchmark at import time. On ROCm that makes PyTorch
# request an exhaustive MIOpen kernel search for every new conv shape (minutes per
# shape, repeated on all 4 ranks), which stalls training. Backend flag only.
torch.backends.cudnn.benchmark = False


def get_args_parser():
    parser = argparse.ArgumentParser()

    # data / io
    parser.add_argument("--data-dir", type=str, required=True,
                        help="ImageNet train directory (class subfolders)")
    parser.add_argument("--latents-path", type=str, required=True,
                        help="Precomputed latents .npy (see preprocess_latents.py)")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Output dir for checkpoints/logs/samples")
    parser.add_argument("--img-size", type=int, default=256)

    # architecture
    parser.add_argument("--model", type=str, default="crossflowDiT_B_2",
                        choices=["crossflowDiT_B_2", "crossflowDiT_L_2", "crossflowDiT_XL_2"])
    parser.add_argument("--num-classes", type=int, default=1000)

    # optimization
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--total-steps", type=int, default=400000)
    parser.add_argument("--p-uncond", type=float, default=0.1,
                        help="CFG label-dropout probability")
    parser.add_argument("--time-eps", type=float, default=1e-4,
                        help="Time endpoint clip; raise (e.g. 0.02) if the r/t^2 "
                             "weight makes early training unstable")
    parser.add_argument("--lpips-weight", type=float, default=0.5,
                        help="Weight of the (diagonal) LPIPS perceptual loss")
    parser.add_argument("--lpips-net", type=str, default="vgg", choices=["vgg", "alex"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"],
                        help="bf16 wraps the forward/JVP in autocast (experimental)")

    # runtime
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=5000)

    # in-loop FID/IS evaluation (small-sample monitoring)
    parser.add_argument("--eval-every", type=int, default=20000)
    parser.add_argument("--eval-num-images", type=int, default=10000,
                        help="Sample count for in-loop FID (must divide num_classes)")
    parser.add_argument("--eval-gen-bsz", type=int, default=64)
    parser.add_argument("--eval-cfg-omega", type=float, default=1.0)
    parser.add_argument("--fid-ref", type=str,
                        default="https://raw.githubusercontent.com/LTH14/JiT/refs/heads/main/fid_stats/jit_in{IMAGE_SIZE}_stats.npz",
                        help="Path or URL to FID reference statistics")

    # wandb
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="crossflow")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser


def infinite_loader(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def save_checkpoint(path, model, ema, opt, step, args):
    """Atomic checkpoint write (tmp file + rename)."""
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "args": vars(args),
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def save_sample_grid(ema_model, step, workdir, seed=0):
    """Render a small fixed-class one-step sample grid from the EMA weights."""
    labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], dtype=torch.int64)
    num_rows = 2
    num_cols = len(labels) // num_rows
    imgs = ema_model.generate(
        n_sample=labels.shape[0],
        rng=tu.BatchGenerator(device=dist.local_device(), seeds=torch.arange(labels.shape[0])),
        y=labels,
        cfg_omega=1.0,
    )
    imgs = tu.device_get(imgs)
    imgs = (imgs + 1) / 2
    grid = np.round(np.clip(imgs * 255, 0, 255)).transpose((0, 2, 3, 1))
    hwc = grid.shape[1:]
    grid = np.einsum("rnhwc->rhnwc", grid.reshape(num_rows, num_cols, *hwc)).reshape(
        num_rows * hwc[0], num_cols * hwc[1], hwc[2]
    )
    grid = grid.astype(np.uint8)  # RGB
    os.makedirs(os.path.join(workdir, "samples"), exist_ok=True)
    cv2.imwrite(os.path.join(workdir, "samples", f"sample_{step:07d}.png"), grid[:, :, ::-1])
    return grid


def run_inloop_eval(ema_model, args, seed):
    """Small-sample FID/IS from EMA weights. All ranks sample; rank 0 gets values."""
    tmp_dir = os.path.join(args.workdir, "eval_tmp")
    fid, inception_score = run_evaluate(
        ema_model,
        tmp_dir,
        fid_ref=args.fid_ref.format(IMAGE_SIZE=ema_model.img_size),
        num_samples=args.eval_num_images,
        device_batch_size=args.eval_gen_bsz,
        initial_seed=seed,
        keep_samples=False,
        cfg_omega=args.eval_cfg_omega,
    )
    return fid, inception_score


def main(args):
    dist.initialize()
    rank = dist.process_index()
    world_size = dist.process_count()
    device = dist.local_device()

    if rank == 0:
        os.makedirs(args.workdir, exist_ok=True)
    dist.print0("Arguments:\n{}".format(args).replace(", ", ",\n"))

    assert args.global_batch_size % world_size == 0, \
        "global batch size must be divisible by world size"
    local_batch_size = args.global_batch_size // world_size

    tu.seed(args.seed)

    # ---------------- model / ema / optimizer ----------------
    model = CrossFlow(
        args.model,
        latent_size=args.img_size // 8,
        num_classes=args.num_classes,
        time_eps=args.time_eps,
    )
    model = tu.device_put(model)
    model.train()
    dist.broadcast_parameters(model)

    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    lpips_fn = None
    if args.lpips_weight > 0:
        lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)

    # ---------------- resume ----------------
    start_step = 0
    latest_path = os.path.join(args.workdir, "latest.pt")
    if os.path.isfile(latest_path):
        dist.print0(f"Resuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        # optimizer state was loaded onto CPU; move it back to the device
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        start_step = ckpt["step"]
        dist.print0(f"Resumed at step {start_step}")

    # ---------------- wandb (rank 0) ----------------
    use_wandb = (rank == 0) and (not args.no_wandb)
    if use_wandb:
        # deterministic id derived from workdir so resumed jobs re-attach the run
        run_id = hashlib.md5(os.path.abspath(args.workdir).encode()).hexdigest()[:16]
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow",
            config=vars(args),
        )

    # ---------------- data ----------------
    dataset = LatentImageNetDataset(args.data_dir, args.latents_path, img_size=args.img_size)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    data_iter = infinite_loader(loader, sampler)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else nullcontext()
    )

    # ---------------- training loop ----------------
    dist.print0(f"Start training: {start_step} -> {args.total_steps} "
                f"(local batch {local_batch_size} x {world_size} gpus)")
    running_loss = 0.0
    running_lpips = 0.0
    t0 = time.time()

    for step in range(start_step, args.total_steps):
        x0, z, y = next(data_iter)
        x0 = x0.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        # CFG label dropout
        drop = torch.rand(y.shape[0], device=device) < args.p_uncond
        y = torch.where(drop, torch.full_like(y, args.num_classes), y)

        opt.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss_cf, x_pred, diagonal = model.compute_loss(z, x0, y)

            lpips_loss = torch.zeros((), device=device)
            if lpips_fn is not None and diagonal.any():
                lpips_loss = lpips_fn(
                    x_pred[diagonal].float(), x0[diagonal].float()
                ).mean()
            loss = loss_cf + args.lpips_weight * lpips_loss

        loss.backward()
        dist.all_reduce_gradients(model)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)

        running_loss += loss_cf.item()
        running_lpips += float(lpips_loss)

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.time() - t0
            imgs_per_sec = args.global_batch_size * n / dt
            dist.print0(
                f"step {step + 1}/{args.total_steps} | "
                f"loss_cf {running_loss / n:.4f} | "
                f"lpips {running_lpips / n:.4f} | "
                f"{imgs_per_sec:.1f} img/s"
            )
            if use_wandb:
                wandb.log(
                    {
                        "loss_cf": running_loss / n,
                        "lpips": running_lpips / n,
                        "img_per_sec": imgs_per_sec,
                        "lr": args.lr,
                    },
                    step=step + 1,
                )
            running_loss = 0.0
            running_lpips = 0.0
            t0 = time.time()

        if (step + 1) % args.sample_every == 0 and rank == 0:
            grid = save_sample_grid(ema.ema_model, step + 1, args.workdir)
            if use_wandb:
                wandb.log({"samples": wandb.Image(grid)}, step=step + 1)

        if (step + 1) % args.eval_every == 0:
            dist.barrier()
            fid, inception_score = run_inloop_eval(ema.ema_model, args, seed=step + 1)
            if use_wandb and fid is not None:
                wandb.log(
                    {"fid": fid, "inception_score": inception_score}, step=step + 1
                )
            model.train()
            dist.barrier()

        if (step + 1) % args.ckpt_every == 0:
            dist.barrier()
            if rank == 0:
                save_checkpoint(latest_path, model, ema, opt, step + 1, args)
                save_checkpoint(
                    os.path.join(args.workdir, f"ckpt_{step + 1:07d}.pt"),
                    model, ema, opt, step + 1, args,
                )
                dist.print0(f"Saved checkpoint at step {step + 1}")
            dist.barrier()

    # final checkpoint + done marker
    dist.barrier()
    if rank == 0:
        save_checkpoint(latest_path, model, ema, opt, args.total_steps, args)
        with open(os.path.join(args.workdir, "DONE"), "w") as f:
            f.write(f"done at step {args.total_steps}\n")
        dist.print0("Training complete; wrote DONE marker.")
    if use_wandb:
        wandb.finish()
    dist.barrier()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
