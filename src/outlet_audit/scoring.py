"""In-folder consensus scoring, robust relative outlier statistic, dual gate, support subgroups."""
from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 0.6745  # Iglewicz & Hoaglin constant: MAD -> sigma for normal data


def cosine_matrix(E: np.ndarray) -> np.ndarray:
    """E rows are L2-normalised; NaN rows (unreadable images) propagate NaN."""
    return E @ E.T


def peer_statistic(peers: np.ndarray, method: str = "median", trim_fraction: float = 0.2, min_peers_for_median: int = 3) -> float:
    """Robust summary of one image's similarities to a set of peers (NaN-free input)."""
    if peers.size == 0:
        return np.nan
    if peers.size < min_peers_for_median:
        return float(peers.mean())
    if method == "median":
        return float(np.median(peers))
    if method == "trimmed_mean":
        k = int(np.floor(trim_fraction * peers.size))
        p = np.sort(peers)
        return float(p[k : peers.size - k].mean())
    raise ValueError(method)


def consensus_scores(S: np.ndarray, method: str = "median", trim_fraction: float = 0.2, min_peers_for_median: int = 3) -> np.ndarray:
    """Per-image agreement with its peers: median (or trimmed mean) of cosine to every other image.
    Not distance-to-centroid: the centroid is pulled toward the very outliers we look for, it
    collapses multi-modal folders (exterior + interior) to a point between modes, and cosine to a
    mean vector is not a measure of typical peer agreement. The median tolerates <50% contamination."""
    n = S.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(S[i, i]):
            continue
        peers = np.delete(S[i], i)
        out[i] = peer_statistic(peers[~np.isnan(peers)], method, trim_fraction, min_peers_for_median)
    return out


def modified_z(x: np.ndarray, mad_floor: float) -> np.ndarray:
    """MAD-based modified z-score, NaN-safe. `mad_floor` keeps z finite in near-constant folders."""
    v = x[~np.isnan(x)]
    if v.size == 0:
        return np.full_like(x, np.nan)
    med = np.median(v)
    mad = max(np.median(np.abs(v - med)), mad_floor)
    return MAD_TO_SIGMA * (x - med) / mad


def dual_gate(z: np.ndarray, suspicion: np.ndarray, z_thresh: float, p_thresh: float) -> np.ndarray:
    """Flag only images that are BOTH relative outliers in their folder AND below the absolute
    (calibrated) similarity floor expressed as suspicion >= p_thresh."""
    with np.errstate(invalid="ignore"):
        return (z < -z_thresh) & (suspicion >= p_thresh)


def support_group(S: np.ndarray, c: int, floor: float, min_size: int) -> tuple[list[int], float] | None:
    """Peers of candidate `c` whose similarity to it clears the calibrated floor. If candidate plus
    peers reach `min_size`, the candidate is part of a coherent subgroup (e.g. interior shots) and we
    return (peer indices, median similarity to those peers); otherwise None."""
    row = S[c].copy()
    row[c] = -np.inf
    peers = np.where(row >= floor)[0].tolist()
    if len(peers) + 1 < min_size:
        return None
    return peers, float(np.median(S[c, peers]))
