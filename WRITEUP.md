# Suspicious outlet-photo detection — write-up

**Data.** 159 outlets, 2 042 JPEGs (5–40 per folder, median 12), all 960×1280, EXIF stripped, none corrupt,
44 byte-identical extra copies (all within-outlet), no cross-outlet duplicates. Bangladeshi recharge shops,
Bangla/English signage. Dominant *legitimate* variation: the same shop with its shutter down, at night, or
from inside.

**Identity signal — DINOv2 ViT-B/14 (self-supervised), CLS token, full frame at 294×224.** Outlet identity
is instance-level (this signboard, this doorway). Text-aligned embeddings (CLIP) collapse instances into
concepts, so two recharge shops with the same brand posters land together; perceptual hashes survive only
near-pixel-identical views. The harness confirms the ordering (easy mode, embedding only, best F0.5 over
the z-grid): DINOv2 P 0.24 / R 0.55 / AP 0.15 > CLIP P 0.23 / R 0.34 / AP 0.14 ≫ pHash P 0.14 / R 0.04 / AP 0.07. ViT-B (not -S) for
instance discrimination at ≈0.4 s/image CPU; DINOv3 is gated.

**Scoring — median pairwise cosine to folder peers.** Each image's score is the median of its cosine to every
other deduplicated image in its folder. A centroid is pulled toward the very outliers being searched for and
collapses multi-modal folders (exterior + interior) to a point between modes; the median tolerates <50 %
contamination and needs no clustering at n = 5.

**Rule — dual gate, calibrated without labels.** (1) *Relative*: MAD modified z-score of the consensus within
the folder, candidate if z < −2.5 — or if the absolute posterior is already ≥ 0.5 (folders whose spread swamps
the MAD). (2) *Absolute*: the same statistic is computed for every image against a
random *other* outlet (guaranteed negatives) and against its own folder (relative outliers trimmed); Gaussian
KDEs of the two populations give a likelihood ratio, and with prior P(foreign) = 0.05 the posterior is the
`suspicion_score` — a probability, not a min-max rescale (legit consensus median 0.73 vs foreign 0.39;
pairwise within 0.72 vs cross 0.39). Candidates then get *SIFT + ratio test + RANSAC homography* against
their 3 nearest peers *among images the folder vouches for* (consensus ≥ the support level below) plus the
folder's 2 most typical images; the inlier count is calibrated the same way (legit median 130 inliers vs 9;
counts below RANSAC's 4-match minimum floored to 4, the 0-inlier KDE bin being a coin flip) and multiplied in
as an independent likelihood ratio — this rescues shutter-down photos whose signboard still matches. A candidate with ≥2 peers above the support level (similarity where
LR = 1, 0.60) is a coherent subgroup (interior shots) and its score is multiplied by that subgroup's own
P(foreign) — only if one member itself clears the support level against the folder; a group coherent only with
itself is foreign, not exempt. Flag ⇔ candidate **and** posterior ≥ p_thresh = 0.30. Median consensus < 0.45
marks the folder `review_folder`: several places, no per-image rule can pick the outlet. Policies:
duplicate copies 0.3 (reason names the twin), cross-outlet identical files 0.9, unreadable files 0.5, n = 1
never flagged, n = 2 both flagged only if their single similarity is below the floor. Reasons are templates filled
with these numbers, CLIP zero-shot scene/brand tags (candidate vs folder majority) and EasyOCR (bn+en)
signage-token overlap with peers.

**Validation — synthetic injection + visual audit.** Seeds 0–2; one image from another outlet is transplanted into
50 % of folders: *easy* = random donor, *hard* = donor with the nearest mean embedding (a look-alike shop).
Originals count as legit, so harness precision is a **lower bound**. Operating point z = 2.5, p_thresh = 0.30
(max F0.5 on easy mode) → precision 0.32, recall 0.53, PR-AUC 0.20. Hard mode stays near chance (P 0.12 / R 0.16 / AP 0.07): look-alike shops need signage-level
identity. Real data: 140 flags in 79/159 outlets — 96 content flags + 44 duplicate copies (7 % review load) — plus
12 folders marked for whole-folder review. A visual audit of every folder (`results/audit_visual.csv`): of the 96
content flags 61 are not the outlet, 18 the same shop (shutter / interior / close-up), 17 undecidable → precision
0.64 (0.81 with undecidable as correct); 61 of 91 audit-confirmed foreign images are caught (recall 0.67).
Correcting the harness negatives with that foreign share gives ≈0.75–0.87 at the operating point. The first gate
(0.62 / 0.47 on the same audit) missed groups of 3–4 photos of one wrong shop: geometry let them vouch for each
other (200+ inliers) and the subgroup exemption accepted a self-coherent group. The fix costs 8 shutter-down photos
of the right shop that a sibling shutter photo used to rescue, and the harness lower bound falls 0.41 → 0.32
because those newly flagged real foreign originals count as errors there.

**Trade-offs.** CPU-only: ≈15 min for embeddings + SIFT, ≈10 s per OCR'd image (candidates + 5 peers per
affected folder only), seconds when warm; Colab T4 (`--device cuda`) ≈17 min cold, SIFT still CPU-bound, same flags
as CPU up to two images at the z boundary. MAD-z is deliberately aggressive; the absolute gate and geometry filter.

**Scalability.** Everything expensive is cached by content hash: a new visit costs one embedding (+ SIFT/OCR
only if it becomes a candidate) and an O(n) folder update; folders are independent, so the job is embarrassingly
parallel. For millions of images: GPU batching (≈2 ms/image on a T4), sqlite/LMDB instead of one .npy per hash,
and a FAISS index over all embeddings to catch a photo reused across outlets (today: exact hashes only).

**Limitations.** (i) a group of photos of one wrong shop still evades when the folder's own spread hides it
from the z gate (posterior at consensus 0.22 is only 0.34, see iii) or the embedding rates the wrong shop ≥0.55
(look-alike groceries); the review flag catches the former only below median 0.45;
(ii) identical brand posters give spurious SIFT inliers between different shops (calibration absorbs the
average effect only); (iii) calibration positives include the data's own outliers, flattening the posterior
(max ≈0.76 without geometry), hence the empirical threshold; (iv) an *old* photo of the *right* shop is
undetectable without timestamps; (v) shutter-down / interior / close-up photos of the right shop remain the main
false-positive class (18 of 96 flags); (vi) OCR on noisy Bangla signage is partial, so text is cited only when both
sides have readable tokens.
