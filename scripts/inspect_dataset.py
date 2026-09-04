"""Report dataset facts before modelling: folder sizes, formats, resolutions, EXIF, duplicates, corrupt files.
Usage: python scripts/inspect_dataset.py --data dataset"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from outlet_audit.io import dedup_groups, scan_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--near-dup-hamming", type=int, default=8)
    a = ap.parse_args()
    ds = scan_dataset(a.data)
    sizes = [len(v) for v in ds.values()]
    res, fmts, exif, dt, gps, kb = collections.Counter(), collections.Counter(), 0, 0, 0, []
    corrupt, exact, near, sha_global = [], 0, 0, collections.defaultdict(list)
    for o, recs in ds.items():
        for r in recs:
            kb.append(r.path.stat().st_size // 1024)
            sha_global[r.sha256].append(f"{o}/{r.file_name}")
            if not r.readable:
                corrupt.append(f"{o}/{r.file_name}")
                continue
            with Image.open(r.path) as im:
                fmts[im.format] += 1
                res[im.size] += 1
                ex = im.getexif()
                exif += bool(ex)
                gps += bool(ex and 0x8825 in ex)
                dt += bool(ex and (0x0132 in ex or 0x9003 in ex.get_ifd(0x8769)))
        for g in dedup_groups(recs, a.near_dup_hamming):
            if len(g) > 1:
                shas = {recs[i].sha256 for i in g}
                exact += len(g) - len(shas)
                near += len(shas) - 1
    cross = [v for v in sha_global.values() if len({x.split('/')[0] for x in v}) > 1]
    print(f"outlets: {len(ds)}  images: {sum(sizes)}")
    print(f"folder size min/median/max: {min(sizes)}/{statistics.median(sizes)}/{max(sizes)}; n<=2 folders: {sum(s <= 2 for s in sizes)}")
    print(f"folder size histogram: {sorted(collections.Counter(sizes).items())}")
    print(f"formats: {dict(fmts)}  resolutions: {res.most_common(5)} ({len(res)} distinct)")
    print(f"file KB min/median/max: {min(kb)}/{statistics.median(kb)}/{max(kb)}")
    print(f"EXIF present: {exif}  with datetime: {dt}  with GPS: {gps}")
    print(f"corrupt/unreadable: {len(corrupt)} {corrupt[:5]}")
    print(f"within-outlet extra copies: exact={exact} near(dHash<={a.near_dup_hamming})={near}")
    print(f"cross-outlet exact duplicates: {len(cross)} {cross[:5]}")


if __name__ == "__main__":
    main()
