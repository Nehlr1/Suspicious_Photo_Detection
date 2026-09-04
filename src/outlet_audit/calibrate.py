"""Label-free likelihood-ratio calibration: contrast a statistic's distribution on presumed-legit
samples (within-folder) against guaranteed negatives (cross-outlet) and turn any value into a
probability-like suspicion via Bayes with an explicit prior. Used for both the embedding consensus
statistic and the geometric-inlier statistic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde

LOG_LR_CLIP = 12.0  # ~ e^12: caps the effect of KDE tails where both densities vanish


@dataclass
class LRModel:
    pos: gaussian_kde  # density of the statistic for legit images
    neg: gaussian_kde  # density for foreign images
    prior: float
    lo: float
    hi: float

    def log_lr(self, x) -> np.ndarray:
        """log p_neg(x) - log p_pos(x): positive means 'looks foreign'."""
        x = np.clip(np.atleast_1d(np.asarray(x, dtype=float)), self.lo, self.hi)
        ln = np.log(self.neg(x) + 1e-300)
        lp = np.log(self.pos(x) + 1e-300)
        return np.clip(ln - lp, -LOG_LR_CLIP, LOG_LR_CLIP)

    def posterior(self, x, extra_log_lr=0.0) -> np.ndarray:
        """P(foreign | x) = prior*LR / (prior*LR + 1 - prior). `extra_log_lr` lets independent
        evidence (geometry) be multiplied in naive-Bayes style."""
        logit = np.log(self.prior / (1 - self.prior)) + self.log_lr(x) + np.asarray(extra_log_lr, dtype=float)
        p = 1 / (1 + np.exp(-logit))
        return p if p.size > 1 else float(p[0])

    def solve_floor(self, p_thresh: float, grid: int = 2001) -> float:
        """Largest statistic value whose posterior is still >= p_thresh (the absolute floor)."""
        xs = np.linspace(self.lo, self.hi, grid)
        ok = np.where(self.posterior(xs) >= p_thresh)[0]
        return float(xs[ok[-1]]) if ok.size else float(self.lo)


def fit_lr(pos_samples, neg_samples, prior: float, bw: str | float = "scott") -> LRModel:
    pos = np.asarray(pos_samples, float)
    neg = np.asarray(neg_samples, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    both = np.concatenate([pos, neg])
    return LRModel(gaussian_kde(pos, bw), gaussian_kde(neg, bw), prior, float(both.min()), float(both.max()))
