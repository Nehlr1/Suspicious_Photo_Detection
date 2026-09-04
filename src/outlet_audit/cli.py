"""outlet-audit CLI: `run` scores the dataset and writes results; `evaluate` runs the injection harness."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import evaluate as ev
from .config import load_config, set_config_values
from .embed import ClipEmbedder, embed_records, get_embedder, resolve_device, set_seed
from .explain import OCR, Explainer, Tagger
from .io import scan_dataset
from .pipeline import Scorer, build_sha_index, fit_consensus_lr, folder_consensus, readable_reps, stack_embeddings
from .report import to_records, write_csv, write_html, write_images_csv, write_json
from .verify import GeomVerifier, fit_geometry_lr

log = logging.getLogger("outlet_audit")


def _common(p):
    p.add_argument("--data", required=True, help="dataset root: <data>/<outlet_id>/*.jpg")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--batch-size", type=int, default=None, help="overrides config batch_size")
    p.add_argument("--no-geometry", action="store_true", help="skip SIFT/RANSAC second stage")
    p.add_argument("--log-level", default="INFO")


def _prepare(a, embedder_kind=None):
    cfg = load_config(a.config)
    set_seed(cfg.seed)
    if a.batch_size:
        cfg.batch_size = a.batch_size
    device = resolve_device(a.device)
    log.info("device=%s", device)
    folders = scan_dataset(a.data)
    recs = [r for v in folders.values() for r in v]
    log.info("%d outlets, %d files, %d unreadable", len(folders), len(recs), sum(not r.readable for r in recs))
    kind = embedder_kind or cfg.embedder
    X = embed_records(recs, get_embedder(kind, cfg, device), cfg.cache_dir, cfg.batch_size)
    emb = {r.sha256: x for r, x in zip(recs, X) if r.readable}
    reps = readable_reps(folders, cfg.dedup.near_dup_hamming)
    E = stack_embeddings(reps, emb)
    rng = np.random.default_rng(cfg.seed)
    lr, floor, stats = fit_consensus_lr(reps, E, cfg, rng)
    stats["embedder"] = kind
    geom = geom_lr = None
    if cfg.geometry.enabled and not a.no_geometry:
        geom = GeomVerifier(cfg.geometry, cfg.cache_dir)
        geom_lr, stats["geometry"] = fit_geometry_lr(reps, E, folder_consensus(E, cfg), geom, cfg.geometry.calib_pairs, cfg.gate.prior_outlier_rate, rng)
    return cfg, device, folders, recs, emb, reps, E, lr, floor, stats, geom, geom_lr


def cmd_run(a):
    cfg, device, folders, recs, emb, reps, E, lr, floor, stats, geom, geom_lr = _prepare(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(cfg, emb, lr, floor, geom, geom_lr)
    stats["support_floor"] = scorer.support_floor
    sha_index = build_sha_index(folders)
    fevals = [scorer.score_folder(o, rs, sha_index) for o, rs in folders.items()]
    flagged = [(fe, i) for fe in fevals for i, im in enumerate(fe.images) if im.flagged]
    log.info("%d flagged images in %d/%d outlets", len(flagged), len({fe.outlet for fe, _ in flagged}), len(fevals))
    tags = None
    if not a.no_tags:
        clip = ClipEmbedder(cfg.models.clip, device)
        Xc = embed_records(recs, clip, cfg.cache_dir, cfg.batch_size)
        tagger = Tagger(clip, cfg.tags)
        tags = {r.sha256: tagger.tag(x) for r, x in zip(recs, Xc) if r.readable}
    ocr = OCR(cfg.ocr, cfg.cache_dir, gpu=device.type == "cuda") if cfg.ocr.enabled and not a.no_ocr else None
    explainer = Explainer(cfg, tags, ocr)
    reasons = {(fe.outlet, fe.images[i].rec.file_name): explainer.reason(fe, i) for fe, i in flagged}
    records = to_records(fevals, reasons)
    write_json(records, out / cfg.output.json)
    write_csv(records, out / cfg.output.csv)
    write_images_csv(fevals, reasons, out / "results_images.csv")
    kinds = {}
    for fe, i in flagged:
        kinds[fe.images[i].kind] = kinds.get(fe.images[i].kind, 0) + 1
    stats["flag_summary"] = {"n_flagged": len(flagged), "n_outlets_with_flags": len({fe.outlet for fe, _ in flagged}), "by_kind": kinds}
    (out / "calibration.json").write_text(json.dumps(stats, indent=2))
    if a.report:
        write_html(fevals, reasons, stats, out / cfg.output.html)
    log.info("wrote %s", ", ".join(str(p) for p in sorted(out.iterdir())))


def cmd_evaluate(a):
    cfg, device, folders, recs, emb, reps, E, lr, floor, stats, geom, geom_lr = _prepare(a, a.embedder)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = np.round(np.concatenate([np.arange(0.02, 0.2, 0.02), np.arange(0.2, 1.0, 0.05)]), 2)
    res = ev.run_harness(folders, reps, E, emb, cfg, lr, floor, geom, geom_lr, thresholds)
    res["calibration"] = stats
    keys = [f"z={z}|{cfg.harness.select_mode}" for z in cfg.harness.z_grid]
    key, best = max(((k, ev.best_threshold(res["runs"][k], cfg.harness.beta)) for k in keys), key=lambda kb: (kb[1][f"f{cfg.harness.beta}"], kb[1]["precision"]))
    z_best = float(key.split("|")[0][2:])
    res["operating_point"] = {"run": key, "z_thresh": z_best, **best}
    tag = f"{a.embedder or cfg.embedder}{'_nogeom' if a.no_geometry or not cfg.geometry.enabled else ''}"
    (out / f"eval_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"\n## {tag}  (pooled over seeds={list(cfg.harness.seeds)} modes={list(cfg.harness.modes)}, z_thresh={cfg.gate.z_thresh})")
    for k, r in res["runs"].items():
        print(f"{k}: n={r['n_scored']} foreign={r['n_foreign']} AP_raw={r['ap_raw']:.3f} AP_gated={r['ap_gated']:.3f}")
    print(ev.format_table(res["runs"][key], cfg.harness.beta))
    print(f"best F{cfg.harness.beta}: z_thresh={z_best} threshold={best['threshold']:.2f} precision={best['precision']:.3f} recall={best['recall']:.3f}")
    if a.write_threshold:
        set_config_values(a.config, {"z_thresh": z_best, "p_thresh": float(best["threshold"])})
        print(f"wrote gate.z_thresh={z_best} gate.p_thresh={best['threshold']:.2f} to {a.config}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="outlet-audit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score every outlet folder and write results.json / results.csv")
    _common(r)
    r.add_argument("--report", action="store_true", help="also write an HTML contact sheet of flagged images")
    r.add_argument("--no-ocr", action="store_true")
    r.add_argument("--no-tags", action="store_true")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("evaluate", help="synthetic-injection harness: precision/recall/PR-AUC vs threshold")
    _common(e)
    e.add_argument("--embedder", choices=["dinov2", "clip", "phash"], default=None, help="override config embedder (baselines)")
    e.add_argument("--write-threshold", action="store_true", help="store the best F-beta (z_thresh, p_thresh) into the config")
    e.set_defaults(fn=cmd_evaluate)
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    a.fn(a)


if __name__ == "__main__":
    main()
