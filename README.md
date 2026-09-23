# CrossFlow (PyTorch, ImageNet)

A from-scratch PyTorch **training** reimplementation of **CrossFlow: One-Step
Generation Across Latent and Pixel Spaces** for class-conditional ImageNet
256×256, following the code style of the sibling `pMF_torch` / `imeanflow_torch`
repos. Built and intended to run on **AMD MI210 (ROCm)** GPUs on a Slurm cluster
with a 24-hour wall-clock limit.

## What CrossFlow does

The network `F_θ` takes a **noised latent** `z_t` (4×32×32 SD-VAE latent) plus a
target time `r`, and directly outputs a **pixel** image (3×256×256). Training
uses the differential cross-space objective (paper eq. 11'), estimating `dF/dt`
along the conditional velocity `ε − z` with a forward-mode JVP and a stop-grad:

```
R_θ = (r/t²)(F_θ − x0) + (1 − r/t) · sg( JVP[F_θ; z_t; (ε − z)] )
```

One-step inference is `x̂ = F_θ(ε, r=0, y)`. Training needs a frozen SD-VAE
**encoder** (to build `z = E(x0)`); inference uses **no VAE** at all.

See `../crossflow/CrossFlow-论文解读与训练伪代码.md` for the full derivation.

## Layout

```
crossflow.py            CrossFlow wrapper: time sampling, JVP loss, one-step generate
train.py                training loop (manual DDP, EMA, checkpoint/resume)
preprocess_latents.py   offline SD-VAE latent precomputation (flip + unflip)
evaluate.py             sample / evaluate (FID+IS) CLI
models/crossflowDiT.py  cross-space DiT (latent patch-embed in, pixel unpatch out)
models/embedder.py      patch / timestep / label embedders   (shared style)
models/torch_models.py  TorchLinear/Embedding/RMSNorm/SwiGLU  (shared style)
utils/                  dist, JAX->torch shims, VAE encoder, dataset, EMA, FID wrapper
scripts/                Slurm launchers (precompute, self-resubmitting train chain, eval)
```

## Design choices (differ from the MeanFlow siblings)

- **Cross-space**: latent input (patch 2 → 16×16 tokens) and pixel output
  (patch 16). Single trunk, single pixel head — no MeanFlow `u/v` dual heads.
- **Conditioning**: in-context prefix tokens for `r` and class label only.
  Standard classifier-free guidance via label dropout (`--p-uncond`), applied on
  the pixel output at sample time (`--cfg-omega`); no MeanFlow interval-guidance
  tokens.
- **RoPE in real arithmetic** (cos/sin) rather than complex ops, so it composes
  with `torch.func.jvp`.
- **No DDP wrapper**: `torch.func.jvp` is incompatible with
  `DistributedDataParallel`, so we broadcast params once and manually all-reduce
  gradients (`utils/torch_dist_util.py`).
- Loss recipe: cross-space L2 residual **+ LPIPS** on the `r = t` reconstruction
  anchors (`--lpips-weight`). GAN/adversarial loss is not included.

## Usage

Install deps (ROCm PyTorch): `pip install -r requirements.txt`.

**1. Precompute latents** (one time):

```bash
sbatch scripts/precompute_latents.sh   # edit DATA_DIR / OUTPUT first
```

**2. Train** (auto-resumes across 24h Slurm blocks):

```bash
sbatch scripts/train_chain.sbatch      # edit the paths block first
```

Each job queues its successor before training; a 24h kill auto-resumes from
`{WORKDIR}/latest.pt`. Training stops (no more resubmits) once
`{WORKDIR}/DONE` is written at `--total-steps`.

**3. Evaluate FID/IS**:

```bash
sbatch scripts/evaluate.sh             # edit CKPT / WORKDIR first
```

Or a quick visual grid on a single GPU:

```bash
python evaluate.py sample --workdir ./vis --ckpt-path /path/latest.pt --model crossflowDiT_B_2
```

## Notes / caveats

- **Precision**: default `--dtype fp32` for the JVP forward (safest on ROCm).
  `--dtype bf16` wraps the forward in autocast for speed but is experimental —
  forward-mode AD + autocast may not compose on all PyTorch/ROCm versions.
- The FID reference uses the JiT ImageNet-256 stats `.npz` (auto-downloaded).
- Model sizes: `crossflowDiT_B_2` (default), `_L_2`, `_XL_2`.
- This is a faithful-in-spirit reproduction; exact paper loss weights, time
  boundaries and the full perceptual+GAN recipe are not claimed to match.
