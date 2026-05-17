# Earth Embeddings for EO: Retrieval, Discovery, and Change-Oriented Search

**Day 2 · Track 2 — Hands-On Session**
**Instructors:** Fuxun Yu, Rishi Madhok (TerraByte AI)

This hands-on session introduces *Earth embeddings* — vector representations of EO products learned by foundation models — and shows how they enable scalable **search, discovery, and comparison** across space, time, and sensors. Participants generate embeddings with pre-trained models (e.g. Google AlphaEarth, IBM–NASA Prithvi, TerraMind, RS-CLIP), build an approximate-nearest-neighbour index, and run metadata-aware similarity search, natural-language retrieval, and change-oriented query composition.

All notebooks are designed to run end-to-end in **Google Colab** (free T4 runtime) within a ~10-minute time-box each. No local installs required.

## Notebooks

| # | Notebook | Topic | Open |
|---|---|---|---|
| 0 | [`notebook_0_eurosat_embeddings_tsne.ipynb`](notebook_0_eurosat_embeddings_tsne.ipynb) | Primer — penultimate-layer embeddings from a ResNet-50 fine-tuned on EuroSAT, nearest-neighbour retrieval sanity check, and 2D t-SNE visualisation of the embedding space. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iceysteel/ESA-NASA-Workshop-2026/blob/main/Day%202/Track%202/Earth-Embeddings-EO/notebook_0_eurosat_embeddings_tsne.ipynb) |
| 1 | [`notebook_1_prithvi_patch_similarity.ipynb`](notebook_1_prithvi_patch_similarity.ipynb) | **Prithvi-EO-2.0 (IBM-NASA, MAE)** — per-patch embeddings from a multispectral + multi-temporal HLS scene over Mexico, cosine-similarity heatmaps from arbitrary query patches. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iceysteel/ESA-NASA-Workshop-2026/blob/main/Day%202/Track%202/Earth-Embeddings-EO/notebook_1_prithvi_patch_similarity.ipynb) |
| 2 | [`notebook_2_dinov3_patch_similarity.ipynb`](notebook_2_dinov3_patch_similarity.ipynb) | **DINOv3 SAT-493M (Meta, self-distillation)** — patch-similarity heatmaps on the *same* Mexico scene rendered as RGB, for side-by-side comparison with Prithvi. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iceysteel/ESA-NASA-Workshop-2026/blob/main/Day%202/Track%202/Earth-Embeddings-EO/notebook_2_dinov3_patch_similarity.ipynb) |
| 3 | [`notebook_3_text_to_image_search.ipynb`](notebook_3_text_to_image_search.ipynb) | **Text → Image search with Git-RSCLIP** — type an English description, get matching satellite images from a 5,400-image EuroSAT pool. Embeddings precomputed and fetched from the HF Hub. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iceysteel/ESA-NASA-Workshop-2026/blob/main/Day%202/Track%202/Earth-Embeddings-EO/notebook_3_text_to_image_search.ipynb) |
| 4 | [`notebook_4_image_to_image_search.ipynb`](notebook_4_image_to_image_search.ipynb) | **Image → Image search with Git-RSCLIP** — same pool, hand the model one satellite image, get visual neighbours. Includes Precision@K benchmark and per-class breakdown. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iceysteel/ESA-NASA-Workshop-2026/blob/main/Day%202/Track%202/Earth-Embeddings-EO/notebook_4_image_to_image_search.ipynb) |
| 5 | [`notebook_5_text_to_change_search.ipynb`](notebook_5_text_to_change_search.ipynb) | **Text → Change search (ChangeCLIP-style)** — type an English description of a change, retrieve matching bi-temporal pairs from a 445-pair LEVIR-CD pool. Uses `normalize(t2 − t1)` as the change embedding and Git-RSCLIP as the (CLIP-style) backbone ChangeCLIP is built on. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iceysteel/ESA-NASA-Workshop-2026/blob/main/Day%202/Track%202/Earth-Embeddings-EO/notebook_5_text_to_change_search.ipynb) |

## Repository assets

- `scripts/build_eurosat_gitrsclip_embeddings.py` — one-time offline script that produces the 5,400-image Git-RSCLIP embedding pool used by Notebooks 3 and 4. Run only when regenerating the pool; participants never execute it.
- `scripts/build_levircd_changeclip_embeddings.py` — one-time offline script that produces the 445-pair Git-RSCLIP bi-temporal embedding pool used by Notebook 5 (LEVIR-CD train split). Same execution model — never run at workshop time.
- Precomputed embedding pools hosted on the Hugging Face Hub:
  - [`zterrabyte/eurosat-gitrsclip-embeddings`](https://huggingface.co/datasets/zterrabyte/eurosat-gitrsclip-embeddings) — 7.7 MB, 5,400 × 768 (EuroSAT image pool for NB3/NB4).
  - [`zterrabyte/levircd-changeclip-embeddings`](https://huggingface.co/datasets/zterrabyte/levircd-changeclip-embeddings) — 1.3 MB, 445 × 2 × 768 (LEVIR-CD bi-temporal pool for NB5).

  Notebooks 3, 4, and 5 pull these at workshop time with `hf_hub_download` (a few seconds on Colab).

## Prerequisites

- Basic Python.
- A Google account for Colab. (`Runtime → Change runtime type → T4 GPU` recommended.)
- Familiarity with EO data formats is helpful but not required.

## Learning outcomes

By the end of the session participants will be able to:

1. Generate Earth embeddings from pre-trained EO foundation models and interpret what is (and isn't) preserved in representation space.
2. Build an ANN index and run filtered similarity search to retrieve spatial / temporal analogs.
3. Apply embedding search to event discovery, clustering, anomaly retrieval, and change-candidate surfacing.
4. Implement natural-language retrieval by aligning text queries with EO embeddings.
5. Compose queries via vector arithmetic to express *with/without* and *before/after* intents.
6. Evaluate retrieval quality and recognise common pitfalls (domain shift, cloud/season confounders, resolution mismatch, indexing trade-offs).

## Foundation model landscape

The five EO foundation models worth knowing for this session, and the pragmatic reason for what each notebook does (or doesn't) load:

| Model | Family | Weights | This session |
|---|---|---|---|
| **Prithvi-EO-2.0** (IBM-NASA, MAE) | masked self-supervised | [HF `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | **NB1** |
| **DINOv3 SAT-493M** (Meta, self-distillation) | self-supervised / contrastive | [HF mirror `timm/vit_large_patch16_dinov3.sat493m`](https://huggingface.co/timm/vit_large_patch16_dinov3.sat493m) | **NB2** |
| **Git-RSCLIP** (CLIP variant on Git10M) | multimodal alignment | [HF `lcybuaa/Git-RSCLIP-base`](https://huggingface.co/lcybuaa/Git-RSCLIP-base) | **NB3 / NB4 / NB5** |
| **AlphaEarth Foundations** (Google DeepMind) | multi-sensor distillation | weights **not released**; precomputed 64-d annual embeddings on [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) | discussed; can't be loaded |
| **TerraMind** (IBM / ESA) | any-to-any multimodal generative | [HF org `ibm-esa-geospatial`](https://huggingface.co/ibm-esa-geospatial), [GitHub `IBM/terramind`](https://github.com/IBM/terramind) | discussed; loader (`terratorch`) is heavy enough that we point to it rather than bundle it in NB2 |
