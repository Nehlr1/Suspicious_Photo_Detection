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
