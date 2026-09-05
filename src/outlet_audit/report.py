"""results.json / results.csv (exact per-outlet schema), a flat per-image CSV, and an HTML contact sheet."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image

from .pipeline import FolderEval
from .verify import top_peers


def to_records(fevals: list[FolderEval], reasons: dict[tuple[str, str], str]) -> list[dict]:
    recs = []
    for fe in fevals:
        order = fe.ranking()
        flagged = [{"file_name": fe.images[i].rec.file_name, "suspicion_score": round(float(fe.images[i].suspicion), 4),
                    "reason": reasons[(fe.outlet, fe.images[i].rec.file_name)]} for i in order if fe.images[i].flagged]
        recs.append({"outlet_id": fe.outlet, "total_images": len(fe.images), "flagged_images": flagged,
                     "ranking": [fe.images[i].rec.file_name for i in order],
                     "folder_median_consensus": None if fe.median_s != fe.median_s else round(fe.median_s, 3), "review_folder": fe.review})
    return recs


def write_json(records: list[dict], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def write_csv(records: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["outlet_id", "total_images", "n_flagged", "flagged_images", "ranking", "folder_median_consensus", "review_folder"])
        for r in records:
            w.writerow([r["outlet_id"], r["total_images"], len(r["flagged_images"]),
                        json.dumps(r["flagged_images"], ensure_ascii=False), json.dumps(r["ranking"]),
                        "" if r["folder_median_consensus"] is None else r["folder_median_consensus"], int(r["review_folder"])])


def write_images_csv(fevals: list[FolderEval], reasons: dict, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["outlet_id", "file_name", "rank", "suspicion_score", "flagged", "kind", "consensus", "z", "reason"])
        for fe in fevals:
            for rank, i in enumerate(fe.ranking(), 1):
                im = fe.images[i]
                w.writerow([fe.outlet, im.rec.file_name, rank, f"{im.suspicion:.4f}", int(im.flagged), im.kind,
                            f"{im.ev.get('s', float('nan')):.3f}", f"{im.ev.get('z', float('nan')):.2f}",
                            reasons.get((fe.outlet, im.rec.file_name), "")])


def _thumb(path: Path, size: int) -> str:
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def write_html(fevals: list[FolderEval], reasons: dict, stats: dict, path: Path, thumb: int = 220, max_rows: int = 300) -> None:
    rows = []
    for fe in fevals:
        for i, im in enumerate(fe.images):
            if im.flagged:
                rows.append((im.suspicion, fe, i))
    rows.sort(key=lambda t: -t[0])
    review = [fe.outlet for fe in fevals if fe.review]
    parts = ["<!doctype html><meta charset='utf-8'><title>Flagged outlet photos</title>",
             "<style>body{font-family:system-ui;margin:20px}table{border-collapse:collapse}td{vertical-align:top;padding:8px;border-bottom:1px solid #ddd}",
             "img{display:block;max-width:100%}.s{font-size:20px;font-weight:bold}.r{max-width:520px}.k{color:#666;font-size:12px}</style>",
             f"<h1>{len(rows)} flagged images across {sum(1 for fe in fevals if any(im.flagged for im in fe.images))}/{len(fevals)} outlets</h1>",
             f"<p><b>{len(review)} outlets below the folder review floor</b> (median consensus &lt; {stats.get('folder_review', {}).get('floor', '?')}; per-image flags unreliable, review the folder): {html.escape(', '.join(review))}</p>" if review else "",
             f"<p class='k'>calibration: {html.escape(json.dumps({k: v for k, v in stats.items() if not isinstance(v, dict)}))}</p>",
             "<table><tr><th>candidate</th><th>nearest peers</th><th>evidence</th></tr>"]
    for score, fe, i in rows[:max_rows]:
        im = fe.images[i]
        peers = []
        if fe.S is not None and i in fe.scored_idx:
            k = fe.scored_idx.index(i)
            peers = [fe.images[fe.scored_idx[j]].rec for j in top_peers(fe.S[k], 2, exclude=k)]
        else:
            peers = [x.rec for x in fe.images if x.rec is not im.rec][:2]
        pimg = "".join(f"<div><img src='{_thumb(p.path, thumb)}'><span class='k'>{html.escape(p.file_name)}</span></div>" for p in peers)
        ev = html.escape(json.dumps({k: v for k, v in im.ev.items() if k not in ("floor", "support_floor", "n_scored")}, ensure_ascii=False, default=str))
        parts.append(f"<tr><td><img src='{_thumb(im.rec.path, thumb)}'><b>{html.escape(fe.outlet)}</b><br>{html.escape(im.rec.file_name)}<br><span class='s'>{score:.2f}</span> <span class='k'>{im.kind}</span></td>"
                     f"<td style='display:flex;gap:6px'>{pimg}</td><td class='r'>{html.escape(reasons[(fe.outlet, im.rec.file_name)])}<br><span class='k'>{ev}</span></td></tr>")
    parts.append("</table>")
    path.write_text("\n".join(parts), encoding="utf-8")
