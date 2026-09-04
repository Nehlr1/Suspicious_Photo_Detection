# outlet-audit — suspicious outlet-photo detection

Flags images in each outlet folder that do not belong to that outlet's photo history, with a
calibrated `suspicion_score` in [0, 1] and an evidence-based reason. Method summary and numbers:
[WRITEUP.md](WRITEUP.md).

Pipeline: DINOv2 self-supervised ViT embeddings → per-image median pairwise cosine to folder peers
→ dual gate (MAD modified z-score **and** a similarity floor calibrated without labels from
within-folder vs cross-outlet similarities, expressed as a likelihood-ratio posterior) → SIFT+RANSAC
geometric verification and coherent-subgroup check on candidates → CLIP zero-shot tags + EasyOCR
signage text for reasons → `results.json` / `results.csv`.

## Setup

```bash
uv sync --extra dev                 # creates .venv from pyproject.toml / uv.lock
# or: python -m pip install -e ".[dev]"
```

Python ≥ 3.10. CPU is enough: on 4 cores the full dataset takes ≈15 min for embeddings + SIFT and
≈10 s per OCR'd image (only flagged candidates and 5 peers per affected folder are OCR'd, ≈350
images ≈ 1 h cold; `--no-ocr` skips it). Everything is cached, so reruns take seconds plus the new
files. `--device cuda` uses a GPU when present (≈50× faster for embeddings and OCR); `suspicious_photo_detection.ipynb`
runs the same command on a Colab T4 (≈17 min cold, SIFT stays CPU-bound). The first run downloads
`facebook/dinov2-base` (~350 MB), `openai/clip-vit-base-patch32` (~600 MB) and EasyOCR
Bengali+English models (~100 MB) into the Hugging Face / EasyOCR caches.

## Run

```bash
# 1. look at the data first
python scripts/inspect_dataset.py --data dataset

# 2. score every outlet, write results + HTML contact sheet
outlet-audit run --data dataset --out results --config config.yaml --device auto --batch-size 16 --report

# 3. synthetic-injection validation (precision / recall / PR-AUC vs threshold), and baselines
outlet-audit evaluate --data dataset --out results --config config.yaml                       # DINOv2 + geometry
outlet-audit evaluate --data dataset --out results --config config.yaml --no-geometry
outlet-audit evaluate --data dataset --out results --config config.yaml --embedder clip  --no-geometry
outlet-audit evaluate --data dataset --out results --config config.yaml --embedder phash --no-geometry
outlet-audit evaluate --data dataset --out results --config config.yaml --write-threshold      # store best F0.5 threshold in config.yaml

# 4. unit tests (scoring, calibration, gating, dedup, metrics on synthetic vectors)
pytest
```

`uv run outlet-audit …` / `uv run pytest` if you did not activate the venv.

## Outputs (`--out results`)

| file | content |
|---|---|
| `results.json` | one record per outlet: `outlet_id`, `total_images`, `flagged_images[{file_name, suspicion_score, reason}]`, `ranking` (all files, most → least suspicious). Outlets with no flags have `flagged_images: []`. |
| `results.csv` | same records, one row per outlet (`flagged_images` / `ranking` as JSON strings) |
| `results_images.csv` | one row per image: rank, score, flagged, kind, consensus, z, reason |
| `calibration.json` | fitted floor, within/cross similarity quantiles, geometry calibration, flag summary |
| `report.html` | contact sheet: each flagged image next to its two nearest peers, reason and raw evidence |
| `eval_<embedder>[_nogeom].json` | harness results: PR table per threshold, PR-AUC, chosen operating point |
| `audit_manual.csv` | single-annotator thumbnail audit of every content flag (genuine / same-shop / undecidable) |

## Results (this dataset)

Operating point from the harness (`evaluate --write-threshold`): `z_thresh = 2.5`, `p_thresh = 0.30`
→ on easy-mode transplants precision 0.41 / recall 0.50 (lower bound: original images are
counted as legit; a manual audit of the real flags, `results/audit_manual.csv`, puts the corrected precision at
≈0.66–0.69, see WRITEUP.md). Baselines through the same harness: DINOv2 without geometry 0.30 / 0.52, CLIP 0.26 / 0.33,
pHash 0.16 / 0.03. On the real data: 105 flags in 64/159 outlets (61 content flags,
44 duplicate copies). Full tables: `results/eval_*.json`, contact sheet: `results/report.html`.

CPU and GPU runs produce the same flag set up to float noise at the z gate (2 images at z ≈ −2.5). One
fix came out of that comparison: the geometry KDE used to be evaluated at 0 inliers, a bin with ≈1
calibration sample per population, so a GPU rerun that drew a different sample hit the −12 log-LR clip there
and silently rescued two audited true positives. Inlier counts below RANSAC's 4-match minimum are now
floored to 4 (`verify.geom_stat_value`); the local flag set is unchanged, the three affected scores rise.

## Configuration

Everything tunable is in `config.yaml` (model ids, input size, consensus statistic, `z_thresh`,
`mad_floor`, `prior_outlier_rate`, `p_thresh`, subgroup size, duplicate/unreadable policy scores,
geometry, tag prompts, OCR languages, harness seeds/modes). The code holds no dataset paths or
magic numbers. `gate.p_thresh` is the operating point; `evaluate --write-threshold` sets it from the
harness.

## Caching / incremental reruns

Embeddings, SIFT features and OCR output are cached under `cache/` keyed by the **sha256 of the
file bytes** (namespaced by model + preprocessing), so a rerun after adding new visits only
processes the new files. Folder statistics are recomputed from the cache in seconds.

## Layout

```
src/outlet_audit/
  cli.py        run | evaluate
  config.py     YAML → attribute dict
  io.py         scan, safe decode, sha256, dHash, duplicate grouping
  embed.py      DINOv2 / CLIP / pHash embedders, hash-keyed cache, device + seed handling
  scoring.py    cosine matrix, median consensus, modified z, dual gate, support subgroup
  calibrate.py  KDE likelihood ratio → posterior, floor solve (shared by embedding + geometry)
  verify.py     SIFT + ratio test + RANSAC, geometry calibration
  pipeline.py   per-folder scoring (dedup, gates, subgroup, geometry, policies)
  explain.py    CLIP zero-shot tags, EasyOCR, reason templates
  evaluate.py   injection harness, PR metrics
  report.py     JSON / CSV / HTML writers
scripts/inspect_dataset.py   dataset facts
tests/                        pytest on synthetic vectors
```
