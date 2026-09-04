"""End-to-end per-folder scoring shared by `run` and the injection harness."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .calibrate import LRModel, fit_lr
from .io import ImageRec, dedup_groups
from .scoring import consensus_scores, cosine_matrix, modified_z, peer_statistic, support_group
from .verify import GeomVerifier, geom_stat_value, peer_indices

log = logging.getLogger(__name__)


@dataclass
class ImageEval:
    rec: ImageRec
    suspicion: float = 0.0
    flagged: bool = False
    kind: str = "scored"  # scored | pair | duplicate | cross_duplicate | unreadable
    ev: dict = field(default_factory=dict)


@dataclass
class FolderEval:
    outlet: str
    images: list[ImageEval]
    scored_idx: list[int]  # indices into images that entered consensus scoring
    S: np.ndarray | None  # cosine matrix among scored images

    def ranking(self) -> list[int]:
        def key(i):
            z = self.images[i].ev.get("z", 0.0)
            return (-self.images[i].suspicion, z if np.isfinite(z) else 0.0, self.images[i].rec.file_name)

        return sorted(range(len(self.images)), key=key)


def dedup(recs: list[ImageRec], hamming: int) -> tuple[list[int], dict[int, int]]:
    """Representative index per duplicate group (first readable member) and extra_copy -> representative."""
    reps, dup_of = [], {}
    for g in dedup_groups(recs, hamming):
        rep = next((i for i in g if recs[i].readable), g[0])
        reps.append(rep)
        for i in g:
            if i != rep:
                dup_of[i] = rep
    return reps, dup_of


def readable_reps(folders: dict[str, list[ImageRec]], hamming: int) -> dict[str, list[ImageRec]]:
    out = {}
    for o, recs in folders.items():
        reps, _ = dedup(recs, hamming)
        out[o] = [recs[i] for i in reps if recs[i].readable]
    return out


def stack_embeddings(reps: dict[str, list[ImageRec]], emb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {o: np.stack([emb[r.sha256] for r in rs]) for o, rs in reps.items() if rs}


def folder_consensus(E: dict[str, np.ndarray], cfg) -> dict[str, np.ndarray]:
    cons = dict(cfg.consensus)
    return {o: consensus_scores(cosine_matrix(X), **cons) for o, X in E.items()}


def fit_consensus_lr(reps: dict[str, list[ImageRec]], E: dict[str, np.ndarray], cfg, rng: np.random.Generator) -> tuple[LRModel, float, dict]:
    """Legit population: consensus scores of within-folder images that pass the relative gate
    (trims the folder's own outliers). Foreign population: the identical statistic computed for each
    image against a random *other* outlet's images (guaranteed negatives). Also records pairwise-level
    within/cross similarity quantiles for the write-up."""
    cons = dict(cfg.consensus)
    outlets = [o for o in E if len(E[o]) >= 2]
    pos, neg, within_pairs, cross_pairs = [], [], [], []
    for o in outlets:
        S = cosine_matrix(E[o])
        s = consensus_scores(S, **cons)
        z = modified_z(s, cfg.gate.mad_floor)
        pos.extend(s[z >= -cfg.gate.z_thresh].tolist())
        within_pairs.extend(S[np.triu_indices(len(S), 1)].tolist())
        d = o
        while d == o:
            d = outlets[rng.integers(len(outlets))]
        C = E[o] @ E[d].T
        neg.extend(peer_statistic(row, **cons) for row in C)
        cross_pairs.extend(C.ravel().tolist())
    lr = fit_lr(pos, neg, cfg.gate.prior_outlier_rate)
    floor = lr.solve_floor(cfg.gate.p_thresh)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    stats = {
        "n_legit": len(pos), "n_foreign": len(neg), "floor": floor, "p_thresh": cfg.gate.p_thresh,
        "legit_consensus_quantiles": dict(zip(q, np.quantile(pos, q).round(3).tolist())),
        "foreign_consensus_quantiles": dict(zip(q, np.quantile(neg, q).round(3).tolist())),
        "within_pair_quantiles": dict(zip(q, np.quantile(within_pairs, q).round(3).tolist())),
        "cross_pair_quantiles": dict(zip(q, np.quantile(cross_pairs, q).round(3).tolist())),
    }
    log.info("consensus LR: floor=%.3f legit med=%.3f foreign med=%.3f", floor, np.median(pos), np.median(neg))
    return lr, floor, stats


class Scorer:
    def __init__(self, cfg, emb: dict[str, np.ndarray], lr: LRModel, floor: float, geom: GeomVerifier | None = None, geom_lr: LRModel | None = None):
        self.cfg, self.emb, self.lr, self.floor, self.geom, self.geom_lr = cfg, emb, lr, floor, geom, geom_lr
        self.cons = dict(cfg.consensus)
        # Similarity above which the evidence favours "same place" (posterior <= prior, i.e. LR <= 1).
        # Used for subgroup support; deliberately independent of the operating threshold p_thresh.
        self.support_floor = lr.solve_floor(cfg.gate.prior_outlier_rate)

    def score_folder(self, outlet: str, recs: list[ImageRec], sha_index: dict[str, list[tuple[str, str]]] | None = None) -> FolderEval:
        cfg, g, pol = self.cfg, self.cfg.gate, self.cfg.policy
        images = [ImageEval(r) for r in recs]
        rep_idx, dup_of = dedup(recs, cfg.dedup.near_dup_hamming)
        scored = [i for i in rep_idx if recs[i].readable]
        S = None
        if scored:
            E = np.stack([self.emb[recs[i].sha256] for i in scored])
            S = cosine_matrix(E)
            s = consensus_scores(S, **self.cons)
            n = len(scored)
            susp = np.atleast_1d(self.lr.posterior(s))
            if n >= 3:
                z = modified_z(s, g.mad_floor)
                rel = z < -g.z_thresh  # relative gate: selects candidates for verification
            elif n == 2:  # relative gate impossible (one shared similarity): absolute gate only
                z, rel = np.full(2, np.nan), np.ones(2, bool)
            else:
                z, rel = np.full(1, np.nan), np.zeros(1, bool)
            med = float(np.nanmedian(s))
            for k, i in enumerate(scored):
                ev = {"s": float(s[k]), "z": float(z[k]), "folder_median_s": med, "floor": self.floor, "support_floor": self.support_floor, "n_scored": n, "candidate": bool(rel[k])}
                p = float(susp[k])
                if rel[k] and n >= 3:
                    if self.geom is not None:
                        peers = peer_indices(S[k], s, self.geom.peers, self.geom.typical_peers, exclude=k)
                        inl, best = self.geom.statistic(recs[i], [recs[scored[j]] for j in peers])
                        extra = float(self.geom_lr.log_lr(geom_stat_value(inl))[0])
                        ev["geom"] = {"inliers": inl, "peer": best.file_name if best else None, "log_lr": extra}
                        p = float(self.lr.posterior(s[k], extra_log_lr=extra))
                    grp = support_group(S, k, self.support_floor, g.min_subgroup_size)
                    if grp is not None:
                        peers_k, s_sub = grp
                        p_sub = float(self.lr.posterior(s_sub))
                        ev["subgroup"] = {"size": len(peers_k) + 1, "s_sub": s_sub, "p_sub": p_sub, "peers": [recs[scored[j]].file_name for j in peers_k]}
                        p *= p_sub
                images[i].suspicion, images[i].ev = p, ev
                images[i].flagged = bool(rel[k]) and p >= g.p_thresh  # absolute gate on the final posterior
                if n == 2 and images[i].flagged:
                    images[i].kind = "pair"
        for i, j in dup_of.items():
            images[i].kind = "duplicate"
            images[i].ev = {"dup_of": recs[j].file_name, "rep_flagged": images[j].flagged}
            images[i].suspicion = max(pol.duplicate_suspicion, images[j].suspicion)
            images[i].flagged = True
        for i, r in enumerate(recs):
            if not r.readable:
                images[i].kind, images[i].suspicion, images[i].flagged, images[i].ev = "unreadable", pol.unreadable_suspicion, True, {}
            elif sha_index is not None:
                others = [(o, f) for o, f in sha_index.get(r.sha256, []) if o != outlet]
                if others:
                    images[i].kind = "cross_duplicate"
                    images[i].ev = {**images[i].ev, "cross_dup": others[0]}
                    images[i].suspicion = max(images[i].suspicion, pol.cross_outlet_duplicate_suspicion)
                    images[i].flagged = True
        return FolderEval(outlet, images, scored, S)


def build_sha_index(folders: dict[str, list[ImageRec]]) -> dict[str, list[tuple[str, str]]]:
    idx: dict[str, list[tuple[str, str]]] = {}
    for o, recs in folders.items():
        for r in recs:
            idx.setdefault(r.sha256, []).append((o, r.file_name))
    return idx
