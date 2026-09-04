"""Image embedders (DINOv2 primary, CLIP + pHash baselines) with a content-hash keyed cache."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .io import ImageRec, load_rgb

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


class Embedder:
    tag: str  # cache namespace
    dim: int

    def embed_images(self, ims: list[Image.Image]) -> np.ndarray:  # (n, dim) L2-normalised
        raise NotImplementedError


class DinoEmbedder(Embedder):
    """DINOv2 CLS token. Input is resized (not cropped) to image_size so the full frame,
    including top-of-frame signage, is seen."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id: str, image_size: tuple[int, int], device: torch.device):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.hw = tuple(image_size)
        self.tag = f"{model_id.replace('/', '__')}_{self.hw[0]}x{self.hw[1]}_cls"
        self.dim = self.model.config.hidden_size
        self._mean = torch.tensor(self.MEAN).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(self.STD).view(1, 3, 1, 1).to(device)

    def _prep(self, ims):
        arr = np.stack([np.asarray(im.resize((self.hw[1], self.hw[0]), Image.BICUBIC), dtype=np.float32) for im in ims])
        x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device) / 255.0
        return (x - self._mean) / self._std

    @torch.no_grad()
    def embed_images(self, ims):
        out = self.model(pixel_values=self._prep(ims)).pooler_output
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()


class ClipEmbedder(Embedder):
    """CLIP image tower (text-aligned baseline + zero-shot tags). Also exposes text embedding."""

    def __init__(self, model_id: str, device: torch.device):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.device = device
        self.tag = f"{model_id.replace('/', '__')}_squash"
        self.dim = self.model.config.projection_dim
        s = self.proc.image_processor.crop_size
        self._size = {"height": s["height"], "width": s["width"]}

    @torch.no_grad()
    def embed_images(self, ims):
        px = self.proc.image_processor(ims, return_tensors="pt", do_center_crop=False, size=self._size)["pixel_values"]
        out = self.model.get_image_features(pixel_values=px.to(self.device))
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tok = self.proc.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        out = self.model.get_text_features(**tok)
        return torch.nn.functional.normalize(out, dim=-1).cpu().numpy()


class PhashEmbedder(Embedder):
    """64-bit DCT perceptual hash as a +-1 vector, so cosine == 1 - 2*hamming/64 and it
    slots into the same consensus code as a baseline."""

    tag = "phash64"
    dim = 64

    def embed_images(self, ims):
        from scipy.fft import dct

        rows = []
        for im in ims:
            g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float64)
            d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
            bits = (d > np.median(d)).ravel().astype(np.float32) * 2 - 1
            rows.append(bits / np.sqrt(64))
        return np.stack(rows)


def get_embedder(kind: str, cfg, device: torch.device) -> Embedder:
    if kind == "dinov2":
        return DinoEmbedder(cfg.models.dinov2, cfg.image_size, device)
    if kind == "clip":
        return ClipEmbedder(cfg.models.clip, device)
    if kind == "phash":
        return PhashEmbedder()
    raise ValueError(f"unknown embedder {kind}")


def embed_records(recs: list[ImageRec], embedder: Embedder, cache_dir: str | Path, batch_size: int) -> np.ndarray:
    """Rows aligned with recs. Unreadable files get NaN rows. Cache key = sha256 of file bytes."""
    cdir = Path(cache_dir) / embedder.tag
    cdir.mkdir(parents=True, exist_ok=True)
    out = np.full((len(recs), embedder.dim), np.nan, dtype=np.float32)
    todo = []
    for i, r in enumerate(recs):
        if not r.readable:
            continue
        f = cdir / f"{r.sha256}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            todo.append(i)
    log.info("%s: %d cached, %d to embed", embedder.tag, len(recs) - len(todo), len(todo))
    for s in tqdm(range(0, len(todo), batch_size), desc=f"embed[{embedder.tag}]", disable=not todo):
        idx = todo[s : s + batch_size]
        vecs = embedder.embed_images([load_rgb(recs[i].path) for i in idx])
        for i, v in zip(idx, vecs):
            out[i] = v
            np.save(cdir / f"{recs[i].sha256}.npy", v.astype(np.float32))
    return out
