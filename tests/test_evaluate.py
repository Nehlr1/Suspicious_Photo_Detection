import numpy as np

from outlet_audit.evaluate import average_precision, pr_table


def test_pr_table_and_ap_on_toy_scores():
    scores = np.array([0.9, 0.8, 0.4, 0.3, 0.1])
    labels = np.array([1, 0, 1, 0, 0])
    cand = np.ones(5, bool)
    rows = pr_table(scores, labels, cand, thresholds=[0.5, 0.35, 0.0], beta=0.5)
    r = {row["threshold"]: row for row in rows}
    assert r[0.5]["precision"] == 0.5 and r[0.5]["recall"] == 0.5 and r[0.5]["n_flagged"] == 2
    assert r[0.35]["recall"] == 1.0 and np.isclose(r[0.35]["precision"], 2 / 3)
    assert r[0.0]["precision"] == 0.4
    assert np.isclose(average_precision(scores, labels), (1.0 + 2 / 3) / 2)


def test_non_candidates_never_flagged():
    scores = np.array([0.9, 0.9])
    labels = np.array([1, 1])
    rows = pr_table(scores, labels, np.array([True, False]), thresholds=[0.5], beta=1.0)
    assert rows[0]["n_flagged"] == 1 and rows[0]["recall"] == 0.5
