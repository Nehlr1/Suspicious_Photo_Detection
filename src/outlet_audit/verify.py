"""Second-stage geometric verification: SIFT + ratio test + RANSAC homography between a candidate
and its most similar folder peers. A geometric match is strong evidence the candidate shows the same
physical place from a different viewpoint. The inlier statistic is calibrated with the same
likelihood-ratio machinery as the embedding consensus."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .calibrate import LRModel, fit_lr
from .io import ImageRec

log = logging.getLogger(__name__)
MIN_MATCHES_FOR_HOMOGRAPHY = 4


class GeomVerifier:
    def __init__(self, gcfg, cache_dir: str | Path):
        self.max_side = gcfg.max_side
        self.ratio = gcfg.ratio
        self.ransac_thresh = gcfg.ransac_thresh
        self.peers = gcfg.peers
        self.typical_peers = gcfg.typical_peers
        self.sift = cv2.SIFT_create(nfeatures=gcfg.n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.cdir = Path(cache_dir) / f"sift_{gcfg.max_side}_{gcfg.n_features}"
        self.cdir.mkdir(parents=True, exist_ok=True)
        self._feat: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pair: dict[tuple[str, str], int] = {}

    def features(self, rec: ImageRec) -> tuple[np.ndarray, np.ndarray]:
        if rec.sha256 in self._feat:
            return self._feat[rec.sha256]
        f = self.cdir / f"{rec.sha256}.npz"
        if f.exists():
            z = np.load(f)
            out = (z["pts"], z["desc"])
        else:
            im = Image.open(rec.path).convert("L")
            scale = self.max_side / max(im.size)
            if scale < 1:
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
            kps, desc = self.sift.detectAndCompute(np.asarray(im), None)
            pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
            desc = np.zeros((0, 128), np.float32) if desc is None else desc.astype(np.float32)
            np.savez(f, pts=pts, desc=desc)
            out = (pts, desc)
        self._feat[rec.sha256] = out
        return out

    def inliers(self, a: ImageRec, b: ImageRec) -> int:
        key = (a.sha256, b.sha256) if a.sha256 <= b.sha256 else (b.sha256, a.sha256)
        if key in self._pair:
            return self._pair[key]
        (pa, da), (pb, db) = self.features(a), self.features(b)
        n = 0
        if len(da) >= 2 and len(db) >= 2:
            good = [m for m, k in self.bf.knnMatch(da, db, k=2) if m.distance < self.ratio * k.distance]
            if len(good) >= MIN_MATCHES_FOR_HOMOGRAPHY:
                src = pa[[m.queryIdx for m in good]]
                dst = pb[[m.trainIdx for m in good]]
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
                n = int(mask.sum()) if mask is not None else 0
        self._pair[key] = n
        return n

    def statistic(self, rec: ImageRec, peers: list[ImageRec]) -> tuple[int, ImageRec | None]:
        """Max inliers over the given peers (already the top-k most similar by embedding)."""
        best, best_peer = 0, None
        for p in peers:
            n = self.inliers(rec, p)
            if n > best:
                best, best_peer = n, p
        return best, best_peer


def geom_stat_value(inliers: int) -> float:
    """RANSAC needs 4 matches, so 0-3 inliers and 4 both mean "no match". Floor them together:
    the 0 bin holds ~1 calibration sample per side, and the KDE ratio there flips sign between
    otherwise identical runs (CPU vs GPU run: same SIFT counts, 3 outliers rescued by a -12 log-LR)."""
    return float(np.log1p(max(inliers, MIN_MATCHES_FOR_HOMOGRAPHY)))


def top_peers(sims: np.ndarray, k: int, exclude: int | None = None) -> list[int]:
    s = sims.copy()
    if exclude is not None:
        s[exclude] = -np.inf
    order = np.argsort(-s)
    return [int(i) for i in order[:k] if np.isfinite(s[i])]


def peer_indices(sims: np.ndarray, consensus: np.ndarray, k_sim: int, k_typical: int, exclude: int | None = None) -> list[int]:
    """The k_sim most similar peers plus the k_typical highest-consensus ("most typical") images of
    the folder. A shutter-down photo's nearest embeddings are often other shutter photos; the typical
    open-shop view is where the signboard is, so it is always included in the geometric check."""
    out = top_peers(sims, k_sim, exclude)
    c = consensus.copy()
    if exclude is not None:
        c[exclude] = -np.inf
    for j in np.argsort(-c)[:k_typical]:
        j = int(j)
        if np.isfinite(c[j]) and j not in out:
            out.append(j)
    return out


def fit_geometry_lr(reps: dict[str, list[ImageRec]], E: dict[str, np.ndarray], consensus: dict[str, np.ndarray], verifier: GeomVerifier, n_pairs: int, prior: float, rng: np.random.Generator) -> tuple[LRModel, dict]:
    """Calibrate the 'max inliers vs peer set' statistic. Legit population: a random image vs its peer
    set in its own folder. Foreign population: the same image vs the equivalent peer set of a random
    other outlet (guaranteed negatives). `consensus[o]` are the per-image consensus scores of folder o."""
    outlets = [o for o in reps if len(reps[o]) > verifier.peers]
    pos, neg = [], []
    for _ in range(n_pairs):
        o = outlets[rng.integers(len(outlets))]
        i = int(rng.integers(len(reps[o])))
        sims = E[o] @ E[o][i]
        peers = peer_indices(sims, consensus[o], verifier.peers, verifier.typical_peers, exclude=i)
        pos.append(geom_stat_value(verifier.statistic(reps[o][i], [reps[o][j] for j in peers])[0]))
        d = o
        while d == o:
            d = outlets[rng.integers(len(outlets))]
        sims = E[d] @ E[o][i]
        peers = peer_indices(sims, consensus[d], verifier.peers, verifier.typical_peers)
        neg.append(geom_stat_value(verifier.statistic(reps[o][i], [reps[d][j] for j in peers])[0]))
    lr = fit_lr(pos, neg, prior)
    stats = {"n_pairs": n_pairs, "legit_median_inliers": float(np.expm1(np.median(pos))), "foreign_median_inliers": float(np.expm1(np.median(neg))),
             "legit_frac_ge_15": float(np.mean(np.expm1(pos) >= 15)), "foreign_frac_ge_15": float(np.mean(np.expm1(neg) >= 15))}
    log.info("geometry LR: %s", stats)
    return lr, stats
