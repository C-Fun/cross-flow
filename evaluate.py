import argparse
import numpy as np
import os
import shutil

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# On ROCm, benchmark mode makes MIOpen run an exhaustive kernel search per conv
# shape (model + Inception), which can stall for an hour. Backend flag only.
torch.backends.cudnn.benchmark = False

import cv2
import wandb

import utils.torch_dist_util as dist
import utils.torch_util as tu
from utils.fidelity_wrapper import calculate_metrics
from crossflow import CrossFlow
from utils.data_util import DATASET_CONFIGS


def print0(*args, **kwargs):
    if dist.process_index() == 0:
        print(*args, **kwargs)


def run_evaluate(
    model,
    workdir,
    fid_ref,
    num_samples: int = 50000,
    device_batch_size: int = 64,
    initial_seed: int = 0,
    keep_samples: bool = False,
    **inference_kwargs,
):
    model.eval()
    world_size = dist.process_count()
    local_rank = dist.process_index()
    num_steps = num_samples // (device_batch_size * world_size) + 1
    device_bs = device_batch_size

    save_folder = os.path.join(workdir, "fid_outputs")
    if local_rank == 0:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    print0(f"Save to: {save_folder}")

    num_classes = model.num_classes
    assert num_samples % num_classes == 0, \
        "Number of FID samples must be divisible by number of classes"
    class_label_gen_world = np.arange(0, num_classes).repeat(num_samples // num_classes)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print0("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * device_bs * i + local_rank * device_bs
        end_idx = start_idx + device_bs
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.from_numpy(labels_gen).long().cuda()
        sample_idx = world_size * device_bs * i + local_rank * device_bs + torch.arange(device_bs)

        sampled_images = model.generate(
            n_sample=sample_idx.shape[0],
            rng=tu.BatchGenerator(device=dist.local_device(), seeds=sample_idx ^ initial_seed),
            y=labels_gen,
            **inference_kwargs,
        )
        sampled_images = tu.device_get(sampled_images)
        sampled_images = sampled_images.transpose(0, 2, 3, 1)  # b h w c
        sampled_images = (sampled_images + 1) / 2

        for b_id in range(device_bs):
            img_id = sample_idx[b_id].item()
            if img_id >= num_samples:
                break
            gen_img = np.round(np.clip(sampled_images[b_id] * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(
                os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5))),
                gen_img,
            )

    dist.barrier()

    fid, inception_score = None, None
    print0("Calculating FID...")
    if dist.process_index() == 0:
        metrics_dict = calculate_metrics(
            input1=save_folder,
            input2=fid_ref,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        if not keep_samples:
            shutil.rmtree(save_folder)

    dist.barrier()
    print0("FID: {}, Inception Score: {}".format(fid, inception_score))
    return fid, inception_score


def load_weights(model, ckpt_path, use_ema=True):
    """Load our training checkpoint dict (with 'ema'/'model') or a raw state_dict.

    Returns the training step stored in the checkpoint (None for raw state_dicts).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    step = None
    if isinstance(ckpt, dict) and ("ema" in ckpt or "model" in ckpt):
        key = "ema" if (use_ema and "ema" in ckpt) else "model"
        step = ckpt.get("step")
        print0(f"Loading '{key}' weights from checkpoint (step {step})")
        model.load_state_dict(ckpt[key], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    return step


@torch.no_grad()
def make_sample_grid(model, grid_size, cfg_omega, seed=0, min_tile=128):
    """Class-conditional sample grid, one class per row, as an RGB uint8 HWC array.

    Rows use evenly spaced classes (or cycle through them if there are fewer
    classes than rows). Small images (e.g. 32px CIFAR) are nearest-upscaled so
    each tile is at least `min_tile` px for viewing.
    """
    n = grid_size * grid_size
    num_classes = model.num_classes
    if num_classes >= grid_size:
        row_classes = np.linspace(0, num_classes - 1, grid_size).round().astype(np.int64)
    else:
        row_classes = np.arange(grid_size) % num_classes
    labels = torch.from_numpy(np.repeat(row_classes, grid_size))

    imgs = model.generate(
        n_sample=n,
        rng=tu.BatchGenerator(device=dist.local_device(), seeds=torch.arange(n) ^ seed),
        y=labels,
        cfg_omega=cfg_omega,
    )
    imgs = (tu.device_get(imgs) + 1) / 2
    imgs = np.round(np.clip(imgs * 255, 0, 255)).astype(np.uint8).transpose(0, 2, 3, 1)
    h, w, c = imgs.shape[1:]
    grid = imgs.reshape(grid_size, grid_size, h, w, c).transpose(0, 2, 1, 3, 4)
    grid = grid.reshape(grid_size * h, grid_size * w, c)
    scale = max(1, min_tile // h)
    if scale > 1:
        grid = grid.repeat(scale, axis=0).repeat(scale, axis=1)
    return grid, row_classes


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str, choices=["sample", "evaluate"],
                        help="sample (visualize a batch) or evaluate (FID)")
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True, metavar="PATH")
    parser.add_argument("--model", type=str, default="crossflowDiT_B_2",
                        choices=["crossflowDiT_B_2", "crossflowDiT_L_2", "crossflowDiT_XL_2"])
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--use-model-weights", action="store_true",
                        help="Use raw model weights instead of EMA")

    # sampling
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--cfg-omega", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = off)")
    parser.add_argument("--num-images", type=int, default=50000)
    parser.add_argument("--gen-bsz", type=int, default=64)
    parser.add_argument("--save-samples", action="store_true")

    parser.add_argument("--fid-ref", type=str, default=None,
                        help=".npz path/URL or torch-fidelity registered input; defaults per dataset")

    # visualization / wandb
    parser.add_argument("--grid-size", type=int, default=8,
                        help="Sample grid is grid_size x grid_size (one class per row)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="crossflow")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)

    return parser


def main(args):
    dist.initialize()
    print0("Working directory:", args.workdir)
    print0("Arguments:\n{}".format(args).replace(", ", ",\n"))

    if dist.process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)

    tu.seed(0)

    cfg = DATASET_CONFIGS[args.dataset]
    if args.img_size is None:
        args.img_size = cfg["img_size"]
    if args.num_classes is None:
        args.num_classes = cfg["num_classes"]
    if args.fid_ref is None:
        args.fid_ref = cfg["fid_ref"]

    model = CrossFlow(
        args.model,
        latent_size=args.img_size // cfg["latent_downsample"],
        latent_channels=cfg["latent_channels"],
        num_classes=args.num_classes,
        out_patch_size=cfg["out_patch_size"],
    )

    if not os.path.isfile(args.ckpt_path):
        print0(f"No checkpoint found at {args.ckpt_path}, exiting.")
        return
    step = load_weights(model, args.ckpt_path, use_ema=not args.use_model_weights)
    model = tu.device_put(model)

    use_wandb = (dist.process_index() == 0) and (not args.no_wandb)
    if use_wandb:
        run_tag = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt_path)))
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="eval",
            name=args.wandb_run_name or f"eval-{run_tag}-step{step}-cfg{args.cfg_omega}",
            config={**vars(args), "ckpt_step": step},
        )

    fid, inception_score = None, None
    if args.mode == "evaluate":
        fid, inception_score = run_evaluate(
            model,
            args.workdir,
            fid_ref=args.fid_ref.format(IMAGE_SIZE=model.img_size),
            num_samples=args.num_images,
            device_batch_size=args.gen_bsz,
            initial_seed=args.sample_seed,
            keep_samples=args.save_samples,
            cfg_omega=args.cfg_omega,
        )

    # Sample grid (both modes), rendered on rank 0 with the same cfg as the eval.
    if dist.process_index() == 0:
        grid, row_classes = make_sample_grid(
            model, args.grid_size, args.cfg_omega, seed=args.sample_seed
        )
        save_path = os.path.join(
            args.workdir, f"samples_{args.grid_size}x{args.grid_size}_cfg{args.cfg_omega}.png"
        )
        cv2.imwrite(save_path, grid[:, :, ::-1])  # RGB -> BGR for cv2
        print0(f"Sample grid saved to {save_path} (rows = classes {row_classes.tolist()})")

        if use_wandb:
            log = {
                "cfg_omega": args.cfg_omega,
                "samples": wandb.Image(
                    grid,
                    caption=f"step {step} | cfg {args.cfg_omega} | rows = classes {row_classes.tolist()}",
                ),
            }
            if fid is not None:
                log.update({"fid": fid, "inception_score": inception_score})
                wandb.summary["fid"] = fid
                wandb.summary["inception_score"] = inception_score
            wandb.log(log, step=step or 0)
            wandb.finish()

    dist.barrier()


if __name__ == "__main__":
    main(get_args_parser().parse_args())
