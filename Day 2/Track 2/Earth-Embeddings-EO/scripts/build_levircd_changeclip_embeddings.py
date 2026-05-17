"""Precompute Git-RSCLIP embeddings on LEVIR-CD bi-temporal pairs.

LEVIR-CD (Chen & Shi, 2020) is the canonical building-change-detection
benchmark: 637 bi-temporal 1024x1024 RGB pairs at 0.5 m / pixel from
Google Earth. Buildings occupy enough pixels at this resolution that
Git-RSCLIP can actually *see* them, so a simple difference-vector
retrieval (`normalize(t2_emb - t1_emb)`) gives a usable text-to-change
signal — much better than S2Looking, where buildings are <1 % of a
Sentinel-2 tile.

ChangeCLIP itself uses OpenAI CLIP-ViT-B/16. We swap that for
`lcybuaa/Git-RSCLIP-base` because OpenAI CLIP barely saw satellite data
and the RS-pretrained tower is what makes retrieval work in practice.

Run ONCE locally to produce
    /tmp/levircd_changeclip_pairs.npz
which is uploaded to the HF Hub dataset repo and pulled by notebook 5 at
workshop time.

Outputs (in the npz):
    embeddings : float16 (N, 2, 768)  L2-normalized; [:,0,:] = t1, [:,1,:] = t2
    image_names: list[str] (N,)
    mask_pct   : float32 (N,)         fraction of change_mask > 0
    sampled_indices : int32 (N,)      row indices into the source split
    split      : str                  which LEVIR-CD split was used
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModel, AutoProcessor
from tqdm.auto import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/tmp/levircd_changeclip_pairs.npz"))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n-pairs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available(): args.device = "cuda"
        elif torch.backends.mps.is_available(): args.device = "mps"
        else: args.device = "cpu"
    print(f"device: {args.device}")

    t0 = time.time()
    proc = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP-base")
    model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP-base").eval().to(args.device)
    print(f"Git-RSCLIP loaded in {time.time()-t0:.1f}s")

    ds = load_dataset("EVER-Z/torchange_levircd", split=args.split)
    print(f"LEVIR-CD {args.split}: {len(ds)} pairs")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_pairs, len(ds))
    idx = np.sort(rng.choice(len(ds), size=n, replace=False))
    print(f"sampling {len(idx)} pairs (seed={args.seed})")

    image_names, mask_pct = [], []
    t1_embs, t2_embs = [], []

    t0 = time.time()
    with torch.inference_mode():
        for i in tqdm(range(0, len(idx), args.batch_size), desc="encoding"):
            batch_idx = idx[i:i + args.batch_size]
            rows = [ds[int(j)] for j in batch_idx]
            t1_imgs = [r["t1_image"].convert("RGB") for r in rows]
            t2_imgs = [r["t2_image"].convert("RGB") for r in rows]

            inp1 = proc(images=t1_imgs, return_tensors="pt").to(args.device)
            inp2 = proc(images=t2_imgs, return_tensors="pt").to(args.device)
            e1 = F.normalize(model.vision_model(**inp1).pooler_output.float(), dim=-1)
            e2 = F.normalize(model.vision_model(**inp2).pooler_output.float(), dim=-1)
            t1_embs.append(e1.cpu().numpy().astype(np.float16))
            t2_embs.append(e2.cpu().numpy().astype(np.float16))

            for r in rows:
                image_names.append(r["image_name"])
                m = np.asarray(r["change_mask"])
                mask_pct.append(float((m > 0).mean()))

    t1_arr = np.concatenate(t1_embs, axis=0)
    t2_arr = np.concatenate(t2_embs, axis=0)
    embeddings = np.stack([t1_arr, t2_arr], axis=1)
    mask_pct = np.array(mask_pct, dtype=np.float32)
    image_names = np.array(image_names)

    print(f"encoded {embeddings.shape} in {time.time()-t0:.1f}s on {args.device}")
    print(f"mask_pct: min={mask_pct.min():.4f}  median={np.median(mask_pct):.4f}  max={mask_pct.max():.4f}")

    cos_t1_t2 = (t1_arr.astype(np.float32) * t2_arr.astype(np.float32)).sum(axis=-1)
    print(f"cos(t1,t2): mean={cos_t1_t2.mean():.3f}  min={cos_t1_t2.min():.3f}  max={cos_t1_t2.max():.3f}")

    diff_norm = np.linalg.norm(t2_arr.astype(np.float32) - t1_arr.astype(np.float32), axis=1)
    rho, p = spearmanr(diff_norm, mask_pct)
    print(f"Spearman(||t2-t1||, mask_pct) = {rho:+.3f}  (p={p:.1e})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        image_names=image_names,
        mask_pct=mask_pct,
        sampled_indices=idx.astype(np.int32),
        split=np.array(args.split),
    )
    print(f"saved -> {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
