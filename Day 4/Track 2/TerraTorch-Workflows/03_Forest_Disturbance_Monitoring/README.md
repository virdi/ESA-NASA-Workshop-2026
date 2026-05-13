# forest-fm-workshop

A small, self-contained Sentinel-2 cloud-free patch time-series dataset
for exploring **geospatial foundation-model embeddings** as a forest
disturbance signal.

## Canonical entry point — `workshop.ipynb`

For the **2nd ESA-NASA Workshop on AI Foundation Models for Earth
Observation**, the single notebook to open is
[`workshop.ipynb`](workshop.ipynb). It's a self-paced
narrative covering:

1. **Data exploration** — 200 cloud-free Sentinel-2 sample time series,
   class distribution, dataset-wide PCA of TerraMind tokens, revisit
   cadence.
2. **Same-period intuition** — embedding-space trajectory of one sample
   with an interactive slider, comparing across years at the same
   calendar period.
3. **NRT detection** — train a linear / MLP head on per-frame anomaly
   features (six recipes; FM vs handcrafted-indices baseline) and apply
   it densely across the patch for a temporal sweep.
4. **NRT attribution** — 4-class agent classification (Clear-Cut /
   Thinning / Wildfire / Windthrow) with a 5-fold stratified CV, plus a
   **detection-gated dense attribution map** that intersects the two
   models so off-event pixels render black.

Switch the active example sample (one per class — `EXAMPLE_CLEAR_SID`,
`EXAMPLE_THIN_SID`, `EXAMPLE_FIRE_SID`, `EXAMPLE_WIND_SID`) by editing
one line at the top of each figure section.

> **Code / data separation.** This directory ships **only** the code
> (`workshop.ipynb`, the `s2tutorial/` package, deps). The
> ~15 GB dataset (patches + embeddings + audit) is a separate bundle
> distributed via S3. Workshop attendees download + unzip the bundle
> on their VM and set `WORKSHOP_DATA_ROOT` to its `data/`
> subdirectory — the notebook reads everything from that pointer.

> **Embeddings.** Frozen **TerraMind-small** features (384-d), generated
> offline with TerraTorch. Two trees are shipped alongside `data/`:
>
> - `embeddings/terramind_v1_small/layer_00/{sid}.zarr/embedding` — dense,
>   `(T, 196, 384)` float32. The `196` is the flat 14×14 token grid.
> - `embeddings_center_patch/terramind_v1_small/layer_00/{sid}.zarr/embedding`
>   — centre token only, `(T, 384)`. The labelled centre pixel falls
>   in token index 105 of the 14×14 grid (row 7, col 7).
>
> `s2tutorial.default_emb_root(data_root)` / `default_emb_center_root(...)`
> resolve the paths; env vars `S2T_EMBEDDING_ROOT` and
> `S2T_EMBEDDING_PATCH_CENTER_ROOT` override.

Each sample's annotation chain follows a strict pattern:

> **Undisturbed → single event → (Revegetation | nothing)**

so the time series is a clean `before / event / after` trio.

Events are sampled in three balanced buckets — **Planned** (Clear-Cut,
Thinning, Forestry Mulching), **Wildfire**, and **Windthrow** — with
~equal counts. Other disturbance categories (Flood, Avalanche, Drought,
Biotic) are deliberately excluded.

## Quick start

```bash
# one-time
pip install -r requirements.txt    # or: uv pip sync requirements.txt
# (no editable install needed — `s2tutorial/` sits next to the notebook
#  and is imported directly from the notebook's working directory.)

# run the workshop notebook
jupyter notebook workshop.ipynb
```

The workshop notebook reads its data root from the `WORKSHOP_DATA_ROOT`
environment variable (default: `../data`) and its audit directory from
`WORKSHOP_AUDIT_DIR` (default: `../audit`). Override either to point at
a downloaded S3 bundle:

```bash
export WORKSHOP_DATA_ROOT=/abs/path/to/workshop_data/data
# AUDIT_DIR defaults to <ROOT>/../audit — only set if you keep it
# somewhere other than next to data/.
jupyter notebook workshop.ipynb
```

If `data/` is empty, set `WORKSHOP_DATA_ROOT` to the location of the
unpacked workshop bundle (see the section below for the SageMaker /
S3 patterns).

## Running on SageMaker / S3

The repo and the data are shipped **separately** so the dataset can
live in S3 (the heavy bit) while the code stays in a normal git repo:

1. **Code** — clone this directory on the SageMaker instance and
   `pip install -r requirements.txt`. < 1 MB on disk. The `s2tutorial/`
   package is imported directly from the notebook's working directory,
   so no editable install is needed.
2. **Data** — receive the bundle (~15 GB tree containing only
   `data/`, `embeddings/`, `embeddings_center_patch/`, `audit/`)
   from the dataset producer. Upload to S3 (`aws s3 sync workshop_data
   s3://your-bucket/workshop_data/`) or supply by some other means.
3. **Glue** — on the SageMaker VM, set `WORKSHOP_DATA_ROOT` to the
   `data/` subdirectory of the unzipped bundle (or to an `s3://` URI).
   The notebook auto-discovers `embeddings/`, `embeddings_center_patch/`,
   and `audit/` as siblings of `data/`.

Three deployment patterns are supported without code changes:

### 1. FSx-for-Lustre or Mountpoint-for-S3 (recommended)

Mount the S3 prefix as a local filesystem on instance startup, then
point the env vars at the mount:

```bash
export WORKSHOP_DATA_ROOT=/home/sagemaker-user/data/workshop_data/data
# WORKSHOP_AUDIT_DIR auto-derives to <ROOT>/../audit; override if needed.
jupyter lab
```

No extra deps. Zarr's random-access pattern hits the mount cache
efficiently — best wall-clock performance for the dense sweeps in
Sections 3 and 4.

### 2. Pre-download to local EBS

A lifecycle / startup script does one `aws s3 sync s3://bucket/workshop_data ~/workshop_data`,
then the notebook reads from local disk. Same env vars as pattern 1,
just pointing at the downloaded copy. No extra deps; fastest steady
state but slowest first-run.

### 3. Direct `s3://` URIs (no mount, no pre-download)

`s2tutorial.loader` accepts `s3://` URIs directly via `fsspec`. Set:

```bash
export WORKSHOP_DATA_ROOT=s3://your-bucket/workshop_data/data
# AUDIT_DIR must stay local — keep a copy of <bundle>/audit/ on disk:
export WORKSHOP_AUDIT_DIR=/local/path/to/workshop_data/audit
```

This pattern requires `s3fs` (already pinned in `requirements.txt` as
a project extra) and AWS credentials available in the SageMaker IAM
role. Slower than (1) and (2) because each zarr chunk is fetched on
demand — fine for a workshop where attendees scrub through a handful
of samples, less fine for training over the full 200.

> **Note on the audit `.npz` files.** `numpy.load` requires a seekable
> stream, so the notebook's `np.load(AUDIT_DIR / "...")` calls expect
> a local path. Keep a copy of the bundle's `audit/` directory (< 1 MB)
> on local disk and point `WORKSHOP_AUDIT_DIR` there, even when
> `WORKSHOP_DATA_ROOT` is an `s3://` URI.

### Building the bundle yourself

The bundle assembler lives in the upstream development repo, not in
this workshop-only tree. Ask the dataset producer for the latest
`workshop_data/` archive.

## Repo layout

```
forest-fm-workshop/
├── data/                       (not in git — supplied via workshop_data.zip)
│   ├── samples.parquet, labels.parquet, frames.parquet
│   ├── splits.parquet          train/val/test split file
│   ├── classes.json
│   └── patches/{sample_id}.zarr/   raw S2 patches (one per sample)
├── embeddings/                 (not in git — same bundle)
│   └── terramind_v1_small/layer_00/{sample_id}.zarr/
│       embedding   (T, 196, 384) float32   dense 14×14 token grid
├── embeddings_center_patch/    (not in git — same bundle)
│   └── terramind_v1_small/layer_00/{sample_id}.zarr/
│       embedding   (T,      384) float32   centre token only
├── audit/                      (not in git — same bundle)
│   ├── center_pixel_features.npz   per-sample centre-pixel feats
│   ├── token_pca.npz               dataset-wide token PCA basis
│   └── rgb_stats.json              per-band RGB percentile stats
├── s2tutorial/                 importable Python package
│   ├── __init__.py
│   ├── loader.py               polars + numpy + zarr (local + s3://)
│   ├── _paths.py               URI-aware path helpers
│   ├── viz.py                  matplotlib helpers
│   ├── nrt.py                  phenological-anomaly framework + recipes
│   ├── dense.py                dense token / pixel feature builders
│   ├── models.py               sklearn-like PyTorch heads
│   ├── rgb_stats.py            RGB normalisation stats
│   └── terratorch_datamodule.py  Lightning + TorchGeo datamodule
├── notebooks/
│   ├── workshop.ipynb          self-paced workshop notebook
│   └── assets/                 PNG explainers referenced from markdown
├── requirements.txt            pip-installable env (shared SageMaker pin list)
├── pyproject.toml              installs the s2tutorial package
├── README.md                   this file
└── .gitignore
```

## Data structure

### `samples.parquet` (one row per sample)

| column                | type     | meaning                                                |
|---                    |---       |---                                                     |
| `sample_id`           | uint16   | 0..N-1, shuffled — *not* a geographic ID               |
| `original_sample_id`  | uint16   | sample id in the upstream extended dataset (kept for tracing back issues; carries no geographic meaning on its own) |
| `window_start`        | date     | first calendar day with any obs in scope               |
| `window_end`          | date     | last calendar day with any obs in scope                |

### `labels.parquet` (period chain — multiple rows per sample)

| column           | type   | meaning                                              |
|---               |---     |---                                                   |
| `sample_id`      | uint16 |                                                      |
| `period_idx`     | uint8  | order in the chain (0, 1, 2, …)                      |
| `label`          | uint16 | 3-digit hierarchical code (see `classes.json`)       |
| `start`          | date   | period start                                         |
| `end_evidence`   | date   | last frame with positive evidence for this period    |
| `end_validity`   | date   | end of period validity (may be `null` for the tail)  |
| `is_event`       | bool   | `True` for the single disturbance event              |

Use `s2tutorial.TimeSeries.label_at(date)` to look up the active label
for any frame.

### `frames.parquet` (one row per stored patch)

| column      | type   | meaning                              |
|---          |---     |---                                   |
| `sample_id` | uint16 |                                      |
| `date`      | date   | acquisition date (calendar-sorted)   |

Frames are filtered to **`cloud_frac == 0`** (strict 0% cloud cover
across the 252 × 252 patch) and then **subsampled to ≤ one frame per
calendar month**, so a 9-year window yields ~108 frames per sample
maximum.

### `classes.json`

Integer code → human-readable name. The codes follow a 3-digit
hierarchy:

- hundreds = L1 (1xx Healthy / 2xx Disturbed),
- tens     = L2 (e.g. 21x Planned, 24x Abiotic),
- units    = L3 (specific operation — Clear-Cut, Wildfire, …).

Codes that may appear in this subset:

| code | name                                |
|------|-------------------------------------|
| 110  | Undisturbed Forest                  |
| 121  | Re-veg with trees (after clear-cut) |
| 122  | Re-veg canopy closing               |
| 123  | Re-veg without trees                |
| 211  | Clear-Cut                           |
| 212  | Thinning                            |
| 213  | Forestry Mulching                   |
| 241  | Drought                             |
| 242  | Wildfire                            |
| 243  | Wind                                |
| 244  | Avalanche                           |
| 245  | Flood                               |

### `patches/{sample_id}.zarr/`

Per-sample Zarr v3 group, ZSTD-9 compressed.

| array     | shape              | dtype   | bands / contents                   |
|---        |---                 |---      |---                                 |
| `s2_10m`  | (T, 4, 252, 252)   | uint16  | B02, B03, B04, B08                 |
| `s2_20m`  | (T, 6, 126, 126)   | uint16  | B05, B06, B07, B8A, B11, B12       |
| `s2_60m`  | (T, 2,  42,  42)   | uint16  | B01, B09                           |
| `s2_scl`  | (T,    126, 126)   | uint8   | Sen2Cor Scene Classification       |

Frozen FM embeddings live in two **external zarr trees** alongside
`data/`, not inside the patches/ stores:

```
embeddings/terramind_v1_small/layer_00/{sid}.zarr/embedding
                   (T, 196, 384) float32  — flat 14×14 token grid
embeddings_center_patch/terramind_v1_small/layer_00/{sid}.zarr/embedding
                   (T,      384) float32  — labelled centre token only
```

Each embedding zarr carries `attrs["dates"]` on the array (matching the
patches' `s2_10m.attrs["dates"]`) and `attrs["sample_id"]` on the group.
`TimeSeries.tm_emb(i)` returns the dense view reshaped to `(14, 14, 384)`;
`TimeSeries.tm_emb_centre(i)` returns the centre token directly as
`(384,)`. Both validate the date alignment lazily on first access.

- Reflectance is stored as the **raw uint16 from Sentinel-2 Processing
  Baseline 04.00+ products**, i.e. `DN = reflectance × 10000 + 1000`
  (ESA's `BOA_ADD_OFFSET = -1000`, introduced Jan 2022 to allow
  negative reflectance to survive as a positive integer). To recover
  physical reflectance: `reflectance = (DN − 1000) / 10000`. The
  loader's `as_reflectance=True` does this for you (and keeps NODATA
  pixels at 0). SCL is unaffected by the offset.
- The time axis `T` is **calendar-sorted ascending**.
- Each sensor array carries `attrs["dates"]` (ISO YYYY-MM-DD strings)
  and (where applicable) `attrs["band_order"]`. **All four arrays
  share the same `dates` list** — guaranteed by build-time invariants.
- Group attrs are `{"sample_id": int}` only. No CRS, no transform, no
  lon/lat, no tile, no upstream identifiers.

The loader asserts this layout on first open of every sample.

## Loader API

```python
from pathlib import Path
import s2tutorial as s2t

ROOT = Path('data')
meta = s2t.load_metadata(ROOT)         # samples + labels + frames + classes

ts = s2t.get_sample(ROOT, sample_id=0, metadata=meta)
print(ts.dates[:5], len(ts), ts.window_start, ts.window_end)

# Pixel access — returns numpy arrays
x10 = ts.s2_10m(0)                      # raw uint16 (4, 252, 252)
x10 = ts.s2_10m(0, as_reflectance=True) # float32 reflectance, divided by 10 000
scl = ts.s2_scl(0)                      # uint8 (126, 126)
emb = ts.tm_emb(0)                      # float32 (14, 14, 384) — dense FM tokens
ctr = ts.tm_emb_centre(0)               # float32 (384,)        — centre token only

# Date-aware label lookup
print(ts.label_at(ts.dates[0]))         # most-recent period.start <= date

# Iterate every sample lazily
for ts in s2t.iter_samples(ROOT):
    ...

# Visualise
s2t.sample_timeline(ts, draw_patches=True, n_patches=8)
s2t.class_breakdown(meta, level=2)
```

## TerraTorch / Lightning datamodule

For training pipelines and reproducible experiments,
`s2tutorial.ForestDisturbanceEmbeddingDataModule` wraps the embedding
zarrs into a Lightning + TorchGeo `NonGeoDataModule`. The same data,
served through the production abstraction:

```python
import s2tutorial as s2t

dm = s2t.ForestDisturbanceEmbeddingDataModule(
    root="data",
    embedding_root="embeddings/terramind_v1_small/layer_00",
    split_file="data/splits.parquet",
    sample_mode="frames",          # or "sequences", "event_centered"
    label_mode="binary",           # or "grouped" (4-class agent), "none"
    batch_size=8,
    num_workers=4,
)
dm.setup("fit")
batch = next(iter(dm.train_dataloader()))   # dict with image, label, ...
```

Sample modes:
- `frames`           — one frame per item; `image` shape `(196, 384)`.
- `sequences`        — a contiguous window of length `sequence_length`;
  `image` shape `(seq, 196, 384)`.
- `event_centered`   — pre-event and post-event pooled vectors;
  `image` shape `(2, 384)` with `event_window_mode="mean"`.

Label modes:
- `binary`           — `No Event` vs `Event` (in-alert-window).
- `grouped`          — `No Event` / `Planned` / `Wildfire` / `Wind`.
- `none`             — no label key (useful for inference).

## License

Data and code are released for the workshop tutorial. Please do not
redistribute without authorisation.
