"""Prepare ImageNet-1k train split in torchvision ImageFolder layout.

ImageNet is licensed/gated: you must accept its terms and authenticate once
before downloading. This script only automates the mechanical steps.

Output layout (what preprocess_latents.py / train.py expect):
    <out>/<class>/<image>.JPEG          # <class> is a sorted, stable folder name

Two sources:

  hf   Download & export from Hugging Face `ILSVRC/imagenet-1k` (needs a token;
       accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
       then `huggingface-cli login` or set HF_TOKEN). Streamed, so it does not
       need a 150GB local cache, but it re-encodes JPEGs. Class folders are the
       4-digit label index (0000..0999).

  tar  Extract an already-downloaded original `ILSVRC2012_img_train.tar`
       (contains 1000 per-WNID inner tars). Lossless and faster. Class folders
       are the WNIDs (n01440764, ...).

Either layout is fine: precompute and training both read the same ImageFolder,
so class indices stay consistent regardless of the folder naming.

Examples:
    python scripts/download_imagenet.py --mode tar \
        --tar /path/ILSVRC2012_img_train.tar --out /path/imagenet/train
    python scripts/download_imagenet.py --mode hf --out /path/imagenet/train
"""

import argparse
import io
import os
import tarfile


def export_from_hf(out_dir, split="train"):
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)

    for i, example in enumerate(ds):
        label = example["label"]
        image = example["image"]
        class_dir = os.path.join(out_dir, f"{label:04d}")
        os.makedirs(class_dir, exist_ok=True)
        image.convert("RGB").save(
            os.path.join(class_dir, f"{i:08d}.JPEG"), quality=95
        )
        if i % 20000 == 0:
            print(f"exported {i} images...", flush=True)

    print(f"Done. Exported {i + 1} images to {out_dir}")


def extract_from_tar(tar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as outer:
        members = [m for m in outer.getmembers() if m.name.endswith(".tar")]
        print(f"Found {len(members)} per-class tars in {tar_path}")
        for idx, member in enumerate(members):
            wnid = os.path.splitext(os.path.basename(member.name))[0]
            class_dir = os.path.join(out_dir, wnid)
            os.makedirs(class_dir, exist_ok=True)

            data = outer.extractfile(member).read()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r") as inner:
                inner.extractall(class_dir)

            if idx % 50 == 0:
                print(f"extracted {idx}/{len(members)} classes ({wnid})", flush=True)

    print(f"Done. Extracted to {out_dir}")


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hf", "tar"], required=True)
    parser.add_argument("--out", required=True, help="Output ImageFolder train dir")
    parser.add_argument("--tar", default=None,
                        help="Path to ILSVRC2012_img_train.tar (--mode tar)")
    parser.add_argument("--split", default="train")
    return parser


def main(args):
    if args.mode == "hf":
        export_from_hf(args.out, split=args.split)
    else:
        assert args.tar, "--tar is required for --mode tar"
        extract_from_tar(args.tar, args.out)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
