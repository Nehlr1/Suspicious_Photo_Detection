# Outlet-audit: Suspicious outlet-photo detection

Field agents photograph mobile-recharge outlets to prove a visit happened. Some cut corners and
upload a photo of a different shop, an old picture, or something unrelated, just to close the task.
Buried in thousands of legitimate images, those fakes are impractical to catch by hand.

Given one folder per outlet holding all photos ever taken there (no timestamps, no visit order),
this tool flags the images that do not belong to that outlet's photo history, with a calibrated
`suspicion_score` in [0, 1] and an evidence-based reason, so a reviewer looks at a few images
instead of all of them. Method summary and numbers: [WRITEUP.md](WRITEUP.md); how the pieces fit
together: [ARCHITECTURE.md](ARCHITECTURE.md).

Input `dataset/outlet_<id>/image_*.jpg`; this dataset is 159 outlets, 2,042 photos, 5-40 per folder.
Output is one record per outlet, including outlets with nothing flagged.

Pipeline: DINOv2 self-supervised ViT embeddings → per-image median pairwise cosine to folder peers
→ dual gate (MAD modified z-score **and** a similarity floor calibrated without labels from
within-folder vs cross-outlet similarities, expressed as a likelihood-ratio posterior; a posterior ≥ 0.5
is a candidate even when the folder's spread swamps the z gate) → SIFT+RANSAC geometric verification
against the images the folder vouches for, and an anchored coherent-subgroup check, on candidates →
CLIP zero-shot tags + EasyOCR signage text for reasons → `results.json` / `results.csv`, plus a
per-folder `review_folder` flag when a folder is too incoherent for any per-image rule.

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
outlet-audit run --data dataset --out results --config config.yaml --device cpu  --batch-size 16 --report   # force CPU
outlet-audit run --data dataset --out results --config config.yaml --device cuda --batch-size 64 --report   # force GPU (cuda:N for a specific card)

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
| `results.json` | one record per outlet: `outlet_id`, `total_images`, `flagged_images[{file_name, suspicion_score, reason}]`, `ranking` (all files, most → least suspicious). Outlets with no flags have `flagged_images: []`. Two extra keys: `folder_median_consensus` and `review_folder` (median below `gate.folder_review_floor`: the folder holds several places, per-image flags are unreliable, review it whole). |
| `results.csv` | same records, one row per outlet (`flagged_images` / `ranking` as JSON strings, then the two folder columns) |
| `results_images.csv` | one row per image: rank, score, flagged, kind, consensus, z, reason |
| `calibration.json` | fitted floor, within/cross similarity quantiles, geometry calibration, flag summary |
| `report.html` | contact sheet: each flagged image next to its two nearest peers, reason and raw evidence |
| `eval_<embedder>[_nogeom].json` | harness results: PR table per threshold, PR-AUC, chosen operating point |
| `audit_manual.csv` | first-pass thumbnail audit of the 61 content flags before the gate fix (genuine / same-shop / undecidable) |
| `audit_visual.csv` | visual audit of every folder after the gate fix: verdict per flag (`TP` not the outlet / `FP` same shop / `U` undecidable), foreign images still missed (`FN`, `FN?` medium confidence) and incoherent folders (`FOLDER`) |

## Results (this dataset)

Operating point `z_thresh = 2.5`, `p_thresh = 0.30` (set by `evaluate --write-threshold` on the first gate, kept
after the gate fix below). Harness, easy-mode transplants pooled over 3 seeds: precision 0.32 / recall 0.53 /
PR-AUC 0.20. That is a lower bound: every original image counts as legit, and the visual audit shows most flagged
originals are real foreign photos; correcting the harness negatives with the audited foreign share gives ≈0.75–0.87
(WRITEUP.md). Baselines through the same harness, embedding only, at their own best F0.5: DINOv2 0.24 / 0.55,
CLIP 0.23 / 0.34, pHash 0.14 / 0.04. On the real data: 140 flags in 79/159 outlets (96 content flags, 44 duplicate copies)
and 12 outlets marked `review_folder`. Visual audit of every folder (`results/audit_visual.csv`): of the 96 content
flags 61 are not the outlet, 18 are the same shop (shutter down / interior / close-up), 17 undecidable → precision
0.64 strict, 0.81 counting undecidable as correct; 61 of the 91 foreign images the audit could confirm are caught
(recall 0.67). Full tables: `results/eval_*.json`, contact sheet: `results/report.html`.

The first run (61 content flags; precision 0.62, recall 0.47 on the same audit) missed
groups of 3–4 photos of one wrong shop in the same folder. Three causes: (a) geometric verification took a candidate's
3 nearest peers regardless of credibility, so transplanted siblings verified each other with 200+ SIFT inliers;
(b) the coherent-subgroup exemption accepted a group that was coherent only with itself; (c) folders whose spread
swamps the MAD z gate had no candidates at all. Now nearest peers for geometry come from images at or above the
support level, a subgroup must contain such an image, a posterior ≥ `gate.p_abs` = 0.5 is a candidate regardless of
z, and folders with median consensus below `gate.folder_review_floor` = 0.45 are reported for whole-folder review.
Cost: 8 more shutter-down photos of the right shop are flagged; they used to be rescued by SIFT-matching a sibling
shutter photo, which is exactly the evidence a transplanted pair gives. Harness precision drops 0.41 → 0.32 through
the fix because the newly flagged real foreign originals count as errors there.

CPU and GPU runs produce the same flag set up to float noise at the z gate (2 images at z ≈ −2.5). One
fix came out of that comparison: the geometry KDE used to be evaluated at 0 inliers, a bin with ≈1
calibration sample per population, so a GPU rerun that drew a different sample hit the −12 log-LR clip there
and silently rescued two audited true positives. Inlier counts below RANSAC's 4-match minimum are now
floored to 4 (`verify.geom_stat_value`); the local flag set is unchanged, the three affected scores rise.

## Configuration

Everything tunable is in `config.yaml` (model ids, input size, consensus statistic, `z_thresh`,
`mad_floor`, `prior_outlier_rate`, `p_thresh`, `p_abs`, `folder_review_floor`, subgroup size, duplicate/unreadable policy scores,
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