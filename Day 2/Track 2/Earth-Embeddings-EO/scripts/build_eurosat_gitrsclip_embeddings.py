"""Precompute Git-RSCLIP embeddings for the EuroSAT test split.

Run ONCE locally (offline) to produce
    assets/eurosat_test_gitrsclip.npz
which is committed to the repo and loaded by notebooks 3 and 4 at workshop time.

The notebooks themselves never run this — at workshop time embedding the full
5,400-image pool would burn >5 minutes of CPU. The single query encoding the
notebooks do at runtime is ~300 ms and well within the per-notebook budget.

Outputs (in the npz):
    embeddings : float16 (N, 768)   — L2-normalized SigLIP image embeddings
    labels     : int32   (N,)       — EuroSAT class indices
    class_names: list[str] (10,)    — class name lookup
    image_ids  : list[str] (N,)     — original EuroSAT file ids (for traceability)
"""
import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from tqdm.auto import tqdm

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "assets" / "eurosat_test_gitrsclip.npz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None,
                   help="cuda / mps / cpu (auto if omitted)")
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

    # --- model -------------------------------------------------------------
    t0 = time.time()
    proc = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP-base")
    model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP-base").eval().to(args.device)
    print(f"model loaded in {time.time()-t0:.1f}s")

    # --- data --------------------------------------------------------------
    ds = load_dataset("timm/eurosat-rgb", split="test")
    class_names = ds.features["label"].names
    print(f"dataset: {len(ds)} images, classes={class_names}")

    # --- encode ------------------------------------------------------------
    all_emb = []
    all_lbl = np.array(ds["label"], dtype=np.int32)
    image_ids = list(ds["image_id"]) if "image_id" in ds.features else [str(i) for i in range(len(ds))]

    t0 = time.time()
    with torch.inference_mode():
        for i in tqdm(range(0, len(ds), args.batch_size), desc="encoding"):
            chunk = ds[i:i + args.batch_size]
            imgs = [im.convert("RGB") for im in chunk["image"]]
            inputs = proc(images=imgs, return_tensors="pt").to(args.device)
            # Use vision_model + visual_projection path equivalent: full forward needs both
            # modalities, so we drive the vision tower directly and apply the projection.
            vis_out = model.vision_model(**inputs)
            # SigLIP vision_model returns BaseModelOutputWithPooling. The "image_embeds"
            # exposed by full forward is `pooler_output` (no extra projection in SigLIP).
            emb = vis_out.pooler_output
            emb = F.normalize(emb.float(), dim=-1)
            all_emb.append(emb.cpu().numpy().astype(np.float16))
    embeddings = np.concatenate(all_emb, axis=0)
    print(f"encoded {embeddings.shape} in {time.time()-t0:.1f}s on {args.device}")

    # --- sanity check ------------------------------------------------------
    # Compare to full-forward image_embeds on a single sample to confirm equivalence.
    sample = ds[0]
    full_in = proc(text=["satellite"], images=[sample["image"].convert("RGB")],
                   padding="max_length", return_tensors="pt").to(args.device)
    with torch.inference_mode():
        out = model(**full_in)
    ref = F.normalize(out.image_embeds.float(), dim=-1).cpu().numpy()[0]
    ours = embeddings[0].astype(np.float32)
    ours = ours / (np.linalg.norm(ours) + 1e-12)
    cos = float(np.dot(ref / (np.linalg.norm(ref) + 1e-12), ours))
    print(f"sanity check: cosine(pooler vs image_embeds) = {cos:.4f}  (should be ~1.0)")

    # --- save --------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        labels=all_lbl,
        class_names=np.array(class_names),
        image_ids=np.array(image_ids),
    )
    sz_mb = args.out.stat().st_size / 1e6
    print(f"saved -> {args.out}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
