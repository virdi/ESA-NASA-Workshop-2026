"""Regenerate the ChangeTX Notebook 6 retrieval embeddings.

This is the reproducibility script for the workshop artifact. It downloads the
public Hugging Face bundle, unzips the clean pre/post images, loads the
repo-local adapter class plus HF model weights, runs inference, and writes a
new `features.npz` with the same array names used by the notebook:

  - `image_features`
  - `caption_text_features`

Custom model code stays in the workshop repository as `changetx_adapter.py`.
The Hugging Face dataset stores artifacts only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipTokenizer

try:
    from transformers import SiglipImageProcessorPil as SiglipImageProcessor
except ImportError:
    from transformers import SiglipImageProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = SCRIPT_DIR.parent
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

from changetx_adapter import ChangeTXBiTemporalConfig, ChangeTXBiTemporalModel  # noqa: E402


DEFAULT_REPO_ID = "FuxunTB/changetx-text-change-demo"
REQUIRED_FILES = [
    "config.json",
    "features.npz",
    "images.zip",
    "manifest.json",
    "model.safetensors",
    "preprocessor_config.json",
    "samples.jsonl",
    "spiece.model",
    "tokenizer_config.json",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_samples(path: Path, max_samples: int) -> list[dict[str, Any]]:
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if max_samples > 0:
        samples = samples[:max_samples]
    if not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (out_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe zip member: {member.filename}")
        archive.extractall(out_dir)


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for candidate in path.iterdir() if candidate.is_file())


def ensure_images(bundle_dir: Path, work_dir: Path, manifest: dict[str, Any], force: bool) -> Path:
    images_info = manifest["images"]
    image_root = work_dir / "images_unzipped"
    pre_dir = image_root / images_info.get("pre_dir", "images/pre")
    post_dir = image_root / images_info.get("post_dir", "images/post")
    expected = int(images_info.get("count_per_time", 0))

    pre_count = count_images(pre_dir)
    post_count = count_images(post_dir)
    if not force and pre_count == expected and post_count == expected:
        print(f"using existing unzipped images: {image_root}")
        return image_root

    if image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)

    archive_path = bundle_dir / images_info.get("archive_path", "images.zip")
    print(f"unzipping {archive_path.name} -> {image_root}")
    t0 = time.time()
    safe_extract_zip(archive_path, image_root)
    pre_count = count_images(pre_dir)
    post_count = count_images(post_dir)
    if pre_count != expected or post_count != expected:
        raise RuntimeError(f"Unexpected image counts after unzip: pre={pre_count}, post={post_count}, expected={expected}")
    print(f"unzipped {pre_count + post_count} images in {time.time() - t0:.1f}s")
    return image_root


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_model(bundle_dir: Path, device: str) -> tuple[ChangeTXBiTemporalModel, SiglipTokenizer, Any]:
    config = ChangeTXBiTemporalConfig.from_pretrained(str(bundle_dir), local_files_only=True)
    model = ChangeTXBiTemporalModel.from_pretrained(
        str(bundle_dir),
        config=config,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = SiglipTokenizer.from_pretrained(str(bundle_dir), local_files_only=True)
    image_processor = SiglipImageProcessor.from_pretrained(str(bundle_dir), local_files_only=True)
    return model, tokenizer, image_processor


def generate_features(
    samples: list[dict[str, Any]],
    image_root: Path,
    bundle_dir: Path,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model, tokenizer, image_processor = load_model(bundle_dir, device=device)

    image_chunks: list[torch.Tensor] = []
    text_chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(samples), batch_size), desc="generate embeddings", unit="batch"):
        batch = samples[start : start + batch_size]
        pre_images = [load_image(image_root / sample["pre_image_path"]) for sample in batch]
        post_images = [load_image(image_root / sample["post_image_path"]) for sample in batch]
        captions = [sample["caption_change"] for sample in batch]

        pre = image_processor(images=pre_images, return_tensors="pt")["pixel_values"].to(device)
        post = image_processor(images=post_images, return_tensors="pt")["pixel_values"].to(device)
        tokens = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            image_chunks.append(model.encode_pair(pre, post, normalize=True).cpu())
            text_chunks.append(model.encode_text(tokens["input_ids"], tokens.get("attention_mask"), normalize=True).cpu())

    return (
        torch.cat(image_chunks, dim=0).numpy().astype(np.float32),
        torch.cat(text_chunks, dim=0).numpy().astype(np.float32),
    )


def cosine_summary(generated: np.ndarray, cached: np.ndarray) -> str:
    generated = generated.astype(np.float32)
    cached = cached[: len(generated)].astype(np.float32)
    sims = (generated * cached).sum(axis=1)
    return f"min={sims.min():.6f} mean={sims.mean():.6f} max={sims.max():.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--work-dir", type=Path, default=Path("changetx_repro"))
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <work-dir>/features.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--force-unzip", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (args.work_dir / "features.npz")

    print(f"repo_id:    {args.repo_id}")
    print(f"work_dir:   {args.work_dir.resolve()}")
    print(f"output:     {output_path.resolve()}")
    print(f"device:     {args.device}")

    t0 = time.time()
    bundle_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=REQUIRED_FILES,
            max_workers=2,
        )
    )
    print(f"downloaded/resolved bundle in {time.time() - t0:.1f}s: {bundle_dir}")

    manifest = read_json(bundle_dir / "manifest.json")
    samples = read_samples(bundle_dir / manifest["samples"]["path"], max_samples=args.max_samples)
    image_root = ensure_images(bundle_dir, args.work_dir, manifest, force=args.force_unzip)

    image_features, caption_text_features = generate_features(
        samples=samples,
        image_root=image_root,
        bundle_dir=bundle_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_features=image_features,
        caption_text_features=caption_text_features,
    )
    print(f"wrote {output_path} with image_features={image_features.shape}, caption_text_features={caption_text_features.shape}")

    if not args.skip_compare:
        cached = np.load(bundle_dir / manifest["features"]["path"])
        print("cosine vs bundled image_features:", cosine_summary(image_features, cached["image_features"]))
        print(
            "cosine vs bundled caption_text_features:",
            cosine_summary(caption_text_features, cached["caption_text_features"]),
        )


if __name__ == "__main__":
    main()
