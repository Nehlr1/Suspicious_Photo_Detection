from pathlib import Path

import numpy as np

from outlet_audit.io import ImageRec, dedup_groups


def _rec(name, sha, bits):
    h = np.packbits(np.array(bits, dtype=np.uint8))
    return ImageRec("o", name, Path(name), sha, True, h)


def test_dedup_groups_exact_and_near():
    base = [0] * 256
    near = base.copy()
    near[:4] = [1, 1, 1, 1]
    far = base.copy()
    far[:40] = [1] * 40
    recs = [_rec("a", "s1", base), _rec("b", "s1", base), _rec("c", "s2", near), _rec("d", "s3", far)]
    assert dedup_groups(recs, near_dup_hamming=8) == [[0, 1, 2], [3]]
    assert dedup_groups(recs, near_dup_hamming=0) == [[0, 1], [2], [3]]


def test_dedup_groups_skip_unreadable():
    recs = [_rec("a", "s1", [0] * 256), ImageRec("o", "x", Path("x"), "s9", False, None)]
    assert dedup_groups(recs, 8) == [[0], [1]]


# ---------------------------------------------------------------------------------------------
# Folder-level gate behaviour on synthetic embeddings (no geometry)
# ---------------------------------------------------------------------------------------------
from outlet_audit.calibrate import fit_lr
from outlet_audit.config import load_config
from outlet_audit.pipeline import Scorer

ROOT = Path(__file__).resolve().parents[1]


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _folder(rng, clusters, noise=0.05):
    """clusters: list of (tag, centre, n). Distinct random dHashes so nothing is deduplicated."""
    recs, emb = [], {}
    for tag, centre, n in clusters:
        for j in range(n):
            sha = f"{tag}{j}"
            recs.append(ImageRec("o", f"{sha}.jpg", Path(f"{sha}.jpg"), sha, True, np.packbits(rng.integers(0, 2, 256).astype(np.uint8))))
            emb[sha] = _unit(centre + noise * rng.standard_normal(centre.size))
    return recs, emb


def _scorer(emb):
    cfg = load_config(ROOT / "config.yaml")
    rng = np.random.default_rng(0)
    lr = fit_lr(rng.normal(0.75, 0.05, 500), rng.normal(0.35, 0.10, 500), cfg.gate.prior_outlier_rate)
    return cfg, Scorer(cfg, emb, lr, lr.solve_floor(cfg.gate.p_thresh))


def test_coherent_foreign_group_is_flagged_but_bridged_subgroup_is_not():
    rng = np.random.default_rng(0)
    u, v, w = (_unit(rng.standard_normal(64)) for _ in range(3))
    w = _unit(0.55 * u + 0.835 * w)  # "interior" view: cos(w, u) ~ 0.55
    # 10 exterior + 4 transplants from an unrelated shop: coherent among themselves, no bridge to the majority
    recs, emb = _folder(rng, [("legit", u, 10), ("foreign", v, 4)])
    cfg, scorer = _scorer(emb)
    fe = scorer.score_folder("o", recs)
    foreign = [im for im in fe.images if im.rec.sha256.startswith("foreign")]
    assert all(im.flagged for im in foreign)
    assert all(im.ev["subgroup"]["size"] == 4 and not im.ev["subgroup"]["anchored"] for im in foreign)
    assert not any(im.flagged for im in fe.images if im.rec.sha256.startswith("legit"))
    assert not fe.review
    # same interior-like subgroup, but two doorway shots sit between the modes and are not outliers -> exempt
    recs, emb = _folder(rng, [("legit", u, 10), ("interior", w, 3), ("bridge", _unit(u + w), 2)])
    cfg, scorer = _scorer(emb)
    fe = scorer.score_folder("o", recs)
    interior = [im for im in fe.images if im.rec.sha256.startswith("interior")]
    assert all(im.ev["candidate"] and im.ev["subgroup"]["anchored"] and not im.flagged for im in interior)


def test_incoherent_folder_is_sent_to_review_and_absolute_gate_still_fires():
    rng = np.random.default_rng(1)
    centres = [_unit(rng.standard_normal(64)) for _ in range(4)]
    recs, emb = _folder(rng, [(f"c{i}", c, 3) for i, c in enumerate(centres)])
    cfg, scorer = _scorer(emb)
    fe = scorer.score_folder("o", recs)
    assert fe.review and fe.median_s < cfg.gate.folder_review_floor
    z = np.array([im.ev["z"] for im in fe.images])
    assert (np.abs(z) < cfg.gate.z_thresh).all()  # the z gate is blind here: every image is equally "typical"
    assert all(im.ev["candidate"] and im.flagged for im in fe.images)  # p >= p_abs makes them candidates anyway
    recs, emb = _folder(rng, [("legit", centres[0], 10)])
    cfg, scorer = _scorer(emb)
    fe = scorer.score_folder("o", recs)
    assert not fe.review and not any(im.flagged for im in fe.images)
