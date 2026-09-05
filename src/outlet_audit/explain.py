"""Evidence-driven reasons: CLIP zero-shot scene/brand tags, EasyOCR signage text, and the numeric
evidence gathered by the scorer, assembled into templated sentences where every clause cites a number
or a piece of extracted text."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

from .embed import ClipEmbedder
from .io import ImageRec
from .pipeline import FolderEval, ImageEval

log = logging.getLogger(__name__)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Tagger:
    def __init__(self, clip: ClipEmbedder, tcfg):
        self.scene_labels = list(tcfg.scene_prompts)
        self.brand_labels = list(tcfg.brand_prompts)
        self.scene_T = clip.embed_texts([tcfg.template.format(p) for p in tcfg.scene_prompts.values()])
        self.brand_T = clip.embed_texts([tcfg.template.format(p) for p in tcfg.brand_prompts.values()])
        self.scale = float(clip.model.logit_scale.exp())

    def tag(self, v: np.ndarray) -> dict:
        ps, pb = _softmax(self.scale * self.scene_T @ v), _softmax(self.scale * self.brand_T @ v)
        return {"scene": self.scene_labels[int(ps.argmax())], "scene_p": float(ps.max()),
                "brand": self.brand_labels[int(pb.argmax())], "brand_p": float(pb.max())}


class OCR:
    def __init__(self, ocfg, cache_dir: str | Path, gpu: bool):
        import easyocr

        self.reader = easyocr.Reader(list(ocfg.langs), gpu=gpu, verbose=False)
        self.min_conf = ocfg.min_conf
        self.cdir = Path(cache_dir) / f"ocr_{'_'.join(ocfg.langs)}"
        self.cdir.mkdir(parents=True, exist_ok=True)

    def read(self, rec: ImageRec) -> list[dict]:
        f = self.cdir / f"{rec.sha256}.json"
        if f.exists():
            return json.loads(f.read_text())
        try:
            items = [{"text": t, "conf": float(c)} for _, t, c in self.reader.readtext(str(rec.path))]
        except Exception as e:  # OCR failure must never kill the run
            log.warning("OCR failed on %s: %s", rec.path, e)
            items = []
        f.write_text(json.dumps(items, ensure_ascii=False))
        return items

    def tokens(self, items: list[dict]) -> set[str]:
        toks = set()
        for it in items:
            if it["conf"] < self.min_conf:
                continue
            for t in re.split(r"[\s\-,.;:|/()\[\]{}'\"!?]+", it["text"].lower()):
                if len(t) >= 2:
                    toks.add(t)
        return toks

    def snippet(self, items: list[dict], n: int = 3, width: int = 60) -> str:
        best = sorted((it for it in items if it["conf"] >= self.min_conf), key=lambda it: -it["conf"])[:n]
        s = " | ".join(it["text"] for it in best)
        return s if len(s) <= width else s[: width - 1] + "…"


class Explainer:
    def __init__(self, cfg, tags: dict[str, dict] | None, ocr: OCR | None):
        self.cfg, self.tags, self.ocr = cfg, tags, ocr

    def _scene_majority(self, fe: FolderEval, exclude: int):
        labels = [self.tags[fe.images[i].rec.sha256]["scene"] for i in fe.scored_idx if i != exclude]
        brands = [self.tags[fe.images[i].rec.sha256]["brand"] for i in fe.scored_idx if i != exclude]
        if not labels:
            return None
        maj = max(set(labels), key=labels.count)
        bmaj = max(set(brands), key=brands.count)
        return maj, labels.count(maj), len(labels), bmaj

    def _ocr_peers(self, fe: FolderEval, exclude: int) -> list[ImageRec]:
        order = sorted((i for i in fe.scored_idx if i != exclude), key=lambda i: -fe.images[i].ev.get("s", 0))
        return [fe.images[i].rec for i in order[: self.cfg.ocr.peers]]

    def reason(self, fe: FolderEval, idx: int) -> str:
        im = fe.images[idx]
        ev = im.ev
        if im.kind == "unreadable":
            return "File could not be decoded as an image."
        if im.kind == "cross_duplicate":
            o, f = ev["cross_dup"]
            return f"Identical file (content hash match) also submitted under {o}/{f}; the same photo cannot document two outlets."
        if im.kind == "duplicate":
            s = f"Identical copy of {ev['dup_of']} (content hash match); duplicate submission adds no new evidence of a visit."
            return s + " The original copy is itself flagged." if ev.get("rep_flagged") else s
        if im.kind == "pair":
            return (f"Only 2 usable images and they do not match each other (similarity {ev['s']:.2f} < absolute floor {ev['floor']:.2f}); "
                    "cannot determine which one is genuine, review both.")
        parts = [f"Median similarity to peers {ev['s']:.2f} vs folder median {ev['folder_median_s']:.2f} "
                 f"(modified z = {ev['z']:.1f}; absolute floor {ev['floor']:.2f})."]
        if self.tags is not None:
            t = self.tags[im.rec.sha256]
            m = self._scene_majority(fe, idx)
            if m:
                maj, k, n, bmaj = m
                if t["scene"] != maj:
                    parts.append(f"Scene reads as '{t['scene']}' while {k}/{n} peers read as '{maj}'.")
                else:
                    parts.append(f"Same scene type as peers ('{maj}') but not the same place.")
                if t["brand"] != "none" and bmaj != "none" and t["brand"] != bmaj and t["brand_p"] >= self.cfg.tags.min_conf:
                    parts.append(f"Telecom branding reads '{t['brand']}' vs '{bmaj}' on peers.")
        if self.ocr is not None:
            cand = self.ocr.read(im.rec)
            ctok = self.ocr.tokens(cand)
            peer_items = [it for r in self._ocr_peers(fe, idx) for it in self.ocr.read(r)]
            ptok = self.ocr.tokens(peer_items)
            if ctok and ptok:
                parts.append(f"Signage text '{self.ocr.snippet(cand)}' shares {len(ctok & ptok)}/{len(ctok)} tokens with outlet signage ('{self.ocr.snippet(peer_items)}').")
            elif ptok:
                parts.append(f"No readable signage text, while peers show '{self.ocr.snippet(peer_items)}'.")
            elif ctok:
                parts.append(f"Signage text '{self.ocr.snippet(cand)}' has no counterpart on peers (no readable peer signage).")
        if "geom" in ev:
            g = ev["geom"]
            if g["log_lr"] > 0:
                parts.append(f"No geometric match to nearest peers (max {g['inliers']} SIFT inliers).")
            else:
                parts.append(f"Partial geometric match to {g['peer']} ({g['inliers']} inliers) was not enough to clear the floor.")
        if "subgroup" in ev:
            sg = ev["subgroup"]
            if not sg.get("anchored", True):
                parts.append(f"{sg['size'] - 1} other images in this folder resemble it (median similarity {sg['s_sub']:.2f}) but none of them matches the rest of the folder either: a coherent group that is foreign as a whole.")
            else:
                parts.append(f"Note: {sg['size'] - 1} other images in this folder resemble it (median similarity {sg['s_sub']:.2f} > support level {ev['support_floor']:.2f}), but not enough to clear the gate.")
        return " ".join(parts)
