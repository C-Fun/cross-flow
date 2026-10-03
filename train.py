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
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import cv2
import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.torch_dist_util as dist
import utils.torch_util as tu
from utils.data_util import LatentImageNetDataset, PixelDataset, DATASET_CONFIGS, build_cifar
from utils.ema_util import EMA
from utils.perceptual import build_perceptual
from crossflow import CrossFlow
from evaluate import run_evaluate

# evaluate.py turns on cudnn.benchmark at import time. On ROCm that makes PyTorch
# request an exhaustive MIOpen kernel search for every new conv shape (minutes per
# shape, repeated on all 4 ranks), which stalls training. Backend flag only.
torch.backends.cudnn.benchmark = False


def get_args_parser():
    parser = argparse.ArgumentParser()

    # data / io
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="imagenet (SD-VAE latents) or cifar10/cifar100 "
                             "(pixel-space sanity check with identity latents)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="ImageNet train dir (class subfolders), or CIFAR root "
                             "(downloaded there if missing)")
    parser.add_argument("--latents-path", type=str, default=None,
                        help="Precomputed latents .npy (imagenet only; see preprocess_latents.py)")
    parser.add_argument("--workdir", type=str, required=True,
                        help="Output dir for checkpoints/logs/samples")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Defaults to the dataset's native size")

    # architecture
    parser.add_argument("--model", type=str, default="crossflowDiT_B_2",
                        choices=["crossflowDiT_B_2", "crossflowDiT_L_2", "crossflowDiT_XL_2"])
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Defaults to the dataset's class count")

    # optimization
    parser.add_argument("--global-batch-size", type=int, default=256,
                        help="Samples per optimizer step (= micro-batch x gpus x grad-accum)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps per optimizer step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear LR warmup steps (constant afterwards)")
    parser.add_argument("--warmup-epochs", type=float, default=0.0,
                        help="Linear LR warmup in epochs (overrides --warmup-steps)")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--total-steps", type=int, default=400000)
    parser.add_argument("--epochs", type=float, default=None,
                        help="Training length in epochs (overrides --total-steps)")
    parser.add_argument("--p-uncond", type=float, default=0.0,
                        help="CFG label-dropout probability (paper: 0)")
    parser.add_argument("--time-eps", type=float, default=1e-2,
                        help="Time endpoint clip for t and r")
    parser.add_argument("--pixel-loss", type=str, default="l1", choices=["l1", "l2"],
                        help="Pixel loss on the CrossFlow residual (paper: L1)")
    parser.add_argument("--adaptive-p", type=float, default=0.0,
                        help="Per-sample adaptive loss normalization power "
                             "(MeanFlow-style, 1/(||R||+c)^p); 0 disables (paper)")
    parser.add_argument("--perc-net", type=str, default="dinov3",
                        choices=["dinov3", "dinov2", "lpips", "none"],
                        help="Perceptual loss backbone on the corrected prediction (paper: DINOv3-B + Huber)")
    parser.add_argument("--perc-weight", type=float, default=1.0,
                        help="Weight of the perceptual loss (applied to all (t, r) samples)")
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
    parser.add_argument("--fid-ref", type=str, default=None,
                        help="FID reference: .npz path/URL or a torch-fidelity registered "
                             "input (e.g. cifar10-train). Defaults per dataset.")

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
    if ema_model.num_classes >= 1000:
        labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], dtype=torch.int64)
    else:
        labels = torch.arange(8, dtype=torch.int64) % ema_model.num_classes
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
    # resolve dataset-dependent defaults
    cfg = DATASET_CONFIGS[args.dataset]
    if args.img_size is None:
        args.img_size = cfg["img_size"]
    if args.num_classes is None:
        args.num_classes = cfg["num_classes"]
    if args.fid_ref is None:
        args.fid_ref = cfg["fid_ref"]
    dist.print0("Arguments:\n{}".format(args).replace(", ", ",\n"))

    assert args.global_batch_size % (world_size * args.grad_accum) == 0, \
        "global batch size must be divisible by world_size * grad_accum"
    local_batch_size = args.global_batch_size // (world_size * args.grad_accum)

    tu.seed(args.seed)

    # ---------------- model / ema / optimizer ----------------
    model = CrossFlow(
        args.model,
        latent_size=args.img_size // cfg["latent_downsample"],
        latent_channels=cfg["latent_channels"],
        num_classes=args.num_classes,
        time_eps=args.time_eps,
        adaptive_p=args.adaptive_p,
        pixel_loss=args.pixel_loss,
        out_patch_size=cfg["out_patch_size"],
        patch_size=cfg["patch_size"],
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

    perc_fn = None
    if args.perc_weight > 0 and args.perc_net != "none":
        perc_fn = build_perceptual(args.perc_net).to(device)

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
    if args.dataset.startswith("imagenet"):
        assert args.latents_path, "--latents-path is required for imagenet"
        dataset = LatentImageNetDataset(args.data_dir, args.latents_path, img_size=args.img_size)
    else:
        # rank 0 downloads first so 4 ranks don't race on the archive
        if rank == 0:
            build_cifar(args.dataset, args.data_dir, download=True)
        dist.barrier()
        dataset = PixelDataset(build_cifar(args.dataset, args.data_dir, download=False))
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

    # epoch-based schedule (paper: 160 epochs, 5 warmup epochs)
    steps_per_epoch = len(dataset) / args.global_batch_size
    if args.epochs is not None:
        args.total_steps = math.ceil(args.epochs * steps_per_epoch)
    if args.warmup_epochs > 0:
        args.warmup_steps = math.ceil(args.warmup_epochs * steps_per_epoch)
    if use_wandb:
        wandb.config.update(
            {"total_steps": args.total_steps, "warmup_steps": args.warmup_steps,
             "local_batch_size": local_batch_size, "steps_per_epoch": steps_per_epoch},
            allow_val_change=True,
        )

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.dtype == "bf16"
        else nullcontext()
    )

    # ---------------- training loop ----------------
    dist.print0(f"Start training: {start_step} -> {args.total_steps} "
                f"(micro batch {local_batch_size} x {world_size} gpus x {args.grad_accum} accum "
                f"= {args.global_batch_size}; warmup {args.warmup_steps} steps)")
    running_loss = 0.0
    running_perc = 0.0
    t0 = time.time()

    for step in range(start_step, args.total_steps):
        # linear LR warmup, constant afterwards
        lr_now = args.lr * min(1.0, (step + 1) / args.warmup_steps) if args.warmup_steps > 0 else args.lr
        for group in opt.param_groups:
            group["lr"] = lr_now

        opt.zero_grad(set_to_none=True)
        step_loss_cf = 0.0
        step_perc = 0.0
        for _ in range(args.grad_accum):
            x0, z, y = next(data_iter)
            x0 = x0.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            # CFG label dropout (paper default: none)
            if args.p_uncond > 0:
                drop = torch.rand(y.shape[0], device=device) < args.p_uncond
                y = torch.where(drop, torch.full_like(y, args.num_classes), y)

            with autocast_ctx:
                loss_main, x_hat, diagonal, loss_cf = model.compute_loss(z, x0, y)

                # perceptual loss on the reconstruction-compatible prediction for
                # ALL (t, r) samples, no r/t^2 weight (paper App. B.4)
                perc_loss = torch.zeros((), device=device)
                if perc_fn is not None:
                    perc_loss = perc_fn(x_hat, x0)
                loss = (loss_main + args.perc_weight * perc_loss) / args.grad_accum

            loss.backward()
            step_loss_cf += loss_cf.item() / args.grad_accum
            step_perc += float(perc_loss) / args.grad_accum

        dist.all_reduce_gradients(model)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        ema.update(model)

        running_loss += step_loss_cf
        running_perc += step_perc

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            dt = time.time() - t0
            imgs_per_sec = args.global_batch_size * n / dt
            dist.print0(
                f"step {step + 1}/{args.total_steps} | "
                f"loss_cf {running_loss / n:.4f} | "
                f"perc {running_perc / n:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{imgs_per_sec:.1f} img/s"
            )
            if use_wandb:
                wandb.log(
                    {
                        "loss_cf": running_loss / n,
                        "perc_loss": running_perc / n,
                        "img_per_sec": imgs_per_sec,
                        "lr": lr_now,
                    },
                    step=step + 1,
                )
            running_loss = 0.0
            running_perc = 0.0
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
