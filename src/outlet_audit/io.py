"""Dataset scanning, safe image loading, content hashing and duplicate grouping."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageRec:
    outlet: str
    file_name: str
    path: Path
    sha256: str
    readable: bool
    dhash: np.ndarray | None = field(default=None, repr=False)  # 32 bytes (256-bit)


def load_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def dhash(im: Image.Image, size: int = 16) -> np.ndarray:
    """Difference hash: (size x size) bits packed into size*size/8 bytes."""
    g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(a ^ b).sum())


def scan_record(outlet: str, path: Path) -> ImageRec:
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(path) as im:
            im.verify()
        h = dhash(load_rgb(path))
        return ImageRec(outlet, path.name, path, sha, True, h)
    except Exception:
        return ImageRec(outlet, path.name, path, sha, False, None)


def scan_dataset(data_dir: str | Path) -> dict[str, list[ImageRec]]:
    """{outlet_id: [ImageRec, ...]} sorted by outlet then file name. Every file is kept,
    including unreadable ones (they are reported, not silently dropped)."""
    data_dir = Path(data_dir)
    out: dict[str, list[ImageRec]] = {}
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
        if files:
            out[d.name] = [scan_record(d.name, p) for p in files]
    return out


def dedup_groups(recs: list[ImageRec], near_dup_hamming: int) -> list[list[int]]:
    """Union-find over exact sha256 matches and dHash Hamming <= threshold.
    Returns groups of indices (singletons included), each sorted, first index = representative."""
    n = len(recs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_sha: dict[str, int] = {}
    for i, r in enumerate(recs):
        if r.sha256 in by_sha:
            union(i, by_sha[r.sha256])
        else:
            by_sha[r.sha256] = i
    # ponytail: O(n^2) within a folder; folders are <=100s of images so this is fine
    for i in range(n):
        if recs[i].dhash is None:
            continue
        for j in range(i + 1, n):
            if recs[j].dhash is not None and hamming(recs[i].dhash, recs[j].dhash) <= near_dup_hamming:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(sorted(g) for g in groups.values())
