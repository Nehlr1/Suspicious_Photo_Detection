# Suspicious outlet-photo detection

## Data

159 outlets, 2,042 JPEGs, 5 to 40 per folder (median 12), all 960x1280 with EXIF
stripped, so no timestamps. Bangladeshi recharge shops, Bangla and English signage.
Most legitimate variation is the same shop with its shutter down, at night, or shot from inside.

## Embedding and scoring

Outlet identity is instance-level: this signboard, this doorway. CLIP collapses instances into
concepts, so two shops sharing brand posters land together, and perceptual hashes survive only
near-identical views. The pipeline therefore uses the DINOv2 ViT-B/14 CLS token on the full frame at
294x224; embedding-only harness numbers confirm the order: DINOv2 P 0.24 / R 0.55, CLIP 0.23 / 0.34, pHash 0.14
/ 0.04.

Each image scores as the median cosine to its deduplicated folder peers. A centroid gets dragged toward
the outliers we are hunting and collapses an exterior-plus-interior folder to a point between the
modes. The median tolerates under 50% contamination and needs no clustering at n = 5.

## Outlier rule

A dual gate, calibrated without labels. The relative half is a MAD modified z-score of the consensus
inside the folder, candidate if z < -2.5. The absolute half computes the same statistic against a
random other outlet (guaranteed negatives) and against the image's own trimmed folder. Gaussian KDEs of
the two give a likelihood ratio, and a prior P(foreign) of 0.05 turns it into the posterior
`suspicion_score`, a probability rather than a rescale (legit consensus median 0.73 against 0.39
foreign). A posterior of 0.5 on its own also makes a candidate, covering folders whose spread swamps
the MAD. Candidates then get SIFT and RANSAC against the 3 nearest peers the folder vouches for.
Inliers calibrate the same way, median 130 against 9, and multiply in as independent evidence, which
rescues shutter-down photos whose signboard still matches. Two peers above the support level make a
coherent subgroup and discount the score, but only if one member clears that level against the folder.
Flag when an image is a candidate and its posterior is at least 0.30. Median consensus below 0.45 marks
`review_folder`: the folder holds several places and no per-image rule applies.

## Results

Synthetic injection, seeds 0 to 2, one foreign image into half the folders. Easy mode gives precision
0.32, recall 0.53, PR-AUC 0.20, a lower bound since originals all count as legit. Hard mode
with a look-alike donor sits near chance and needs signage-level identity. On real data: 140 flags in
79 of 159 outlets, 96 content flags plus 44 duplicates, a 7% review load, and 12 folders sent for
whole-folder review. A visual audit of all 96 content flags found 61 genuinely not the outlet, 18 the
same shop, 17 undecidable, so precision is 0.64 strict, 0.81 counting undecidable as correct, recall 0.67 against the 91 confirmed foreign images.

## Trade-offs and scalability

A CPU run takes about 15 minutes for embeddings and SIFT plus 10 seconds per OCR'd image, and OCR
touches only candidates and 5 peers per affected folder. A T4 finishes cold in about 17 minutes.
Everything expensive is cached by content hash, so a new visit costs one embedding and an O(n) folder
update, and folders are independent, which makes the job embarrassingly parallel. At millions of images:
GPU batching, sqlite or LMDB in place of one .npy per hash, and a FAISS index to catch photos reused
across outlets, which today only exact hashes catch.

## Limitations

A group of photos of one wrong shop can still hide inside a folder's own spread, or score above 0.55
when the two shops look alike. Identical brand posters produce spurious SIFT inliers between different
shops. Calibration positives include the data's own outliers, flattening the posterior to about 0.76
without geometry and forcing an empirical threshold. An old photo of the right shop is undetectable
without timestamps. Shutter-down and interior shots of the right shop remain the main false-positive
class, 18 of 96 flags. OCR on noisy Bangla signage is partial, so text is cited only when both sides
have readable tokens.
