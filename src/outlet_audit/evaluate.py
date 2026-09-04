"""Synthetic-injection validation harness and PR metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def pr_table(scores: np.ndarray, labels: np.ndarray, candidate: np.ndarray, thresholds, beta: float) -> list[dict]:
    """Precision / recall / F1 / F-beta at each suspicion threshold, respecting the relative gate
    (only `candidate` images can ever be flagged)."""
    rows = []
    pos = labels.sum()
    for t in thresholds:
        flag = candidate & (scores >= t)
        tp = int((flag & (labels == 1)).sum())
        nf = int(flag.sum())
        prec = tp / nf if nf else 0.0
        rec = tp / pos if pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        b2 = beta * beta
        fb = (1 + b2) * prec * rec / (b2 * prec + rec) if prec + rec else 0.0
        rows.append({"threshold": float(t), "precision": prec, "recall": rec, "f1": f1, f"f{beta}": fb, "n_flagged": nf, "tp": tp})
    return rows


# ---------------------------------------------------------------------------------------------
# Synthetic injection harness
# ---------------------------------------------------------------------------------------------
import copy  # noqa: E402
import logging  # noqa: E402
from dataclasses import replace  # noqa: E402

from .io import ImageRec  # noqa: E402
from .pipeline import Scorer  # noqa: E402

log = logging.getLogger(__name__)


def inject(folders: dict[str, list[ImageRec]], reps: dict[str, list[ImageRec]], E: dict[str, np.ndarray], hcfg, mode: str, rng: np.random.Generator):
    """Transplant images between outlets. 'easy' = random donor outlet; 'hard' = the donor whose
    mean embedding is closest to the target (a look-alike shop). Returns (new folders, foreign keys)."""
    outlets = [o for o in folders if o in E]
    means = {o: E[o].mean(0) / np.linalg.norm(E[o].mean(0)) for o in outlets}
    targets = rng.choice(outlets, size=max(1, round(hcfg.inject_fraction * len(outlets))), replace=False)
    new = {o: list(recs) for o, recs in folders.items()}
    foreign = set()
    for t in targets:
        others = [o for o in outlets if o != t]
        if mode == "easy":
            donor = others[rng.integers(len(others))]
        elif mode == "hard":
            donor = max(others, key=lambda o: float(means[o] @ means[t]))
        else:
            raise ValueError(mode)
        have = {r.sha256 for r in new[t]}
        pool = [r for r in reps[donor] if r.sha256 not in have]
        for r in rng.choice(pool, size=min(hcfg.injections_per_folder, len(pool)), replace=False):
            name = f"__inj__{donor}__{r.file_name}"
            new[t].append(replace(r, outlet=t, file_name=name))
            foreign.add((t, name))
    return new, foreign


def _collect(fevals, foreign):
    scores, labels, cand = [], [], []
    for fe in fevals:
        for i in fe.scored_idx:
            im = fe.images[i]
            scores.append(im.suspicion)
            labels.append(int((fe.outlet, im.rec.file_name) in foreign))
            cand.append(bool(im.ev.get("candidate", False)))
    return np.array(scores), np.array(labels), np.array(cand)


def run_harness(folders, reps, E, emb, cfg, lr, floor, geom, geom_lr, thresholds) -> dict:
    h = cfg.harness
    out = {"thresholds": [float(t) for t in thresholds], "runs": {}}
    for z in h.z_grid:
        cfg_z = copy.deepcopy(cfg)
        cfg_z.gate.z_thresh = z
        scorer = Scorer(cfg_z, emb, lr, floor, geom, geom_lr)
        pooled = {m: ([], [], []) for m in h.modes}
        for seed in h.seeds:
            rng = np.random.default_rng(seed)
            for mode in h.modes:
                inj, foreign = inject(folders, reps, E, h, mode, rng)
                fevals = [scorer.score_folder(o, recs, None) for o, recs in inj.items()]
                s, l, c = _collect(fevals, foreign)
                for acc, v in zip(pooled[mode], (s, l, c)):
                    acc.append(v)
                log.info("z=%.1f seed=%d mode=%s: %d foreign among %d scored", z, seed, mode, l.sum(), len(l))
        for mode in list(h.modes) + ["all"]:
            keys = h.modes if mode == "all" else [mode]
            s = np.concatenate([x for m in keys for x in pooled[m][0]])
            l = np.concatenate([x for m in keys for x in pooled[m][1]])
            c = np.concatenate([x for m in keys for x in pooled[m][2]])
            gated = np.where(c, s, 0.0)
            out["runs"][f"z={z}|{mode}"] = {
                "n_scored": int(len(l)), "n_foreign": int(l.sum()),
                "ap_raw": average_precision(s, l), "ap_gated": average_precision(gated, l),
                "table": pr_table(s, l, c, thresholds, h.beta),
            }
    return out


def best_threshold(run: dict, beta: float) -> dict:
    return max(run["table"], key=lambda r: (r[f"f{beta}"], r["precision"]))


def format_table(run: dict, beta: float) -> str:
    lines = [f"| threshold | precision | recall | F1 | F{beta} | flagged |", "|---|---|---|---|---|---|"]
    for r in run["table"]:
        lines.append(f"| {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r[f'f{beta}']:.3f} | {r['n_flagged']} |")
    return "\n".join(lines)
