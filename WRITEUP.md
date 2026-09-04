# Suspicious outlet-photo detection — write-up

**Data.** 159 outlets, 2 042 JPEGs (5–40 per folder, median 12), all 960×1280, EXIF stripped, none corrupt,
44 byte-identical extra copies (all within-outlet), no cross-outlet duplicates. Bangladeshi recharge shops,
Bangla/English signage. Dominant *legitimate* variation: the same shop with its shutter down, at night, or
from inside.

**Identity signal — DINOv2 ViT-B/14 (self-supervised), CLS token, full frame at 294×224.** Outlet identity
is instance-level (this signboard, this doorway). Text-aligned embeddings (CLIP) collapse instances into
concepts, so two recharge shops with the same brand posters land together; perceptual hashes survive only
near-pixel-identical views. The harness confirms the ordering (easy mode, embedding only, best F0.5 over
the z-grid): DINOv2 P 0.30 / R 0.52 / AP 0.19 > CLIP P 0.26 / R 0.33 / AP 0.11 ≫ pHash P 0.16 / R 0.03 / AP 0.05. ViT-B (not -S) for
instance discrimination at ≈0.4 s/image CPU; DINOv3 is gated.

**Scoring — median pairwise cosine to folder peers.** Each image's score is the median of its cosine to every
other deduplicated image in its folder. A centroid is pulled toward the very outliers being searched for,
collapses multi-modal folders (exterior + interior) to a point between modes, and cosine-to-mean does not
measure peer agreement; the median tolerates <50 % contamination and needs no clustering at n = 5.

**Rule — dual gate, calibrated without labels.** (1) *Relative*: MAD modified z-score of the consensus within
the folder, candidate if z < −2.5. (2) *Absolute*: the same statistic is computed for every image against a
random *other* outlet (guaranteed negatives) and against its own folder (relative outliers trimmed); Gaussian
KDEs of the two populations give a likelihood ratio, and with prior P(foreign) = 0.05 the posterior is the
`suspicion_score` — a probability, not a min-max rescale (legit consensus median 0.73 vs foreign 0.39;
pairwise within 0.72 vs cross 0.39). Relative candidates then get *SIFT + ratio test + RANSAC homography*
against their 3 nearest peers plus the folder's 2 most typical images; the inlier count is calibrated the
same way (legit median 130 inliers vs 9; 86 % vs 25 % ≥ 15; counts below RANSAC's 4-match minimum are floored
to 4, since the 0-inlier bin holds ≈1 calibration sample per population and its KDE ratio is a coin flip) and multiplied in as an independent
likelihood ratio — this rescues shutter-down photos whose signboard still matches. A candidate with ≥2 peers
above the support level (similarity where LR = 1, 0.60) is a coherent subgroup and its score is multiplied
by that subgroup's own P(foreign). Flag ⇔ relative outlier **and** posterior ≥ p_thresh = 0.30. Policies:
duplicate copies 0.3 (reason names the twin), cross-outlet identical files 0.9, unreadable files 0.5, n = 1
never flagged, n = 2 both flagged only if their single similarity is below the floor. Reasons are templates filled
with these numbers, CLIP zero-shot scene/brand tags (candidate vs folder majority) and EasyOCR (bn+en)
signage-token overlap with peers.

**Validation — synthetic injection.** Seeds 0–2; one image from another outlet is transplanted into 50 % of
folders: *easy* = random donor (the "random storefront" fraud), *hard* = donor with the nearest mean embedding
(a look-alike shop). Originals count as legit, so harness precision is a **lower bound**. Operating point =
max F0.5 (short review list) on easy mode: z = 2.5, p_thresh = 0.30 → precision 0.41, recall 0.50, PR-AUC 0.24
(embedding only: P 0.30 / R 0.52 / AP 0.19). Hard mode stays near chance (P 0.14 / R 0.13 / AP 0.07): look-alike
shops need signage-level identity. On the real data: 105 flags in 64/159 outlets — 61 content flags + 44 duplicate
copies (5 % review load). A manual audit of the 61 content flags (`results/audit_manual.csv`): 26 clearly not the outlet (2 unrelated
scenes, 19 different shops/places, 5 close-ups that do not show the outlet), 29 same shop (shutter / interior /
new angle = false positives), 6 undecidable. Correcting the harness negatives for that contamination gives an
expected precision of 0.66–0.69 at the operating point (undecidable counted as legit / half).

**Trade-offs.** CPU-only: ≈15 min for embeddings + SIFT, ≈10 s per OCR'd image (candidates + 5 peers per
affected folder only), seconds when warm. Colab T4 (`--device cuda`): ≈17 min cold, SIFT still CPU-bound; CPU and GPU
runs give the same flags except two images at the z = −2.5 boundary (float noise). Before the inlier floor, a GPU rerun that
drew a different calibration sample hit the −12 log-LR clip at 0 inliers and silently rescued two audited true positives. MAD-z is deliberately aggressive; the absolute gate and geometry
filter. F0.5 buys precision at recall ≈0.5.

**Scalability.** Everything expensive is cached by content hash: a new visit costs one embedding (+ SIFT/OCR
only if it becomes a candidate) and an O(n) update of its folder; folders are independent, so the job is
embarrassingly parallel and per-folder O(n²) is negligible. For millions of images: GPU batching (≈2 ms/image on a T4), sqlite/LMDB instead of one .npy per hash, and a FAISS
index over all embeddings to catch a photo reused across outlets (today: exact hashes only).

**Limitations.** (i) ≥3 fraudulent photos of the same wrong shop form a "coherent subgroup" and evade;
(ii) identical brand posters give spurious SIFT inliers between different shops (calibration absorbs the
average effect only); (iii) calibration positives include the data's own outliers, flattening the posterior
(max ≈0.76 without geometry), hence the empirical threshold; (iv) an *old* photo of the *right* shop is
undetectable without timestamps; (v) shutter-down / interior / signboard-close-up photos of the right shop remain the main
false-positive class (≈half of the flags); QR-poster close-ups are scored as legit when the same poster hangs at the shop; (vi) OCR on noisy Bangla signage is partial, so text is cited only when both sides have
readable tokens.
