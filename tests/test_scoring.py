import numpy as np
import pytest

from outlet_audit.scoring import consensus_scores, cosine_matrix, dual_gate, modified_z, support_group


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _cluster(rng, centre, n, noise=0.1):
    return _unit(centre + noise * rng.standard_normal((n, centre.size)))


def test_planted_outlier_gets_lowest_consensus_and_extreme_z():
    rng = np.random.default_rng(0)
    u = _unit(rng.standard_normal(64))
    E = np.vstack([_cluster(rng, u, 9), _unit(rng.standard_normal((1, 64)))])
    s = consensus_scores(cosine_matrix(E))
    assert np.argmin(s) == 9
    z = modified_z(s, mad_floor=0.01)
    assert z[9] < -3.5
    assert (z[:9] > -3.5).all()


def test_median_consensus_more_robust_than_centroid_under_40pct_contamination():
    rng = np.random.default_rng(1)
    u, v = _unit(rng.standard_normal(64)), _unit(rng.standard_normal(64))
    E = np.vstack([_cluster(rng, u, 6, 0.02), _cluster(rng, v, 4, 0.02)])
    s = consensus_scores(cosine_matrix(E))
    c = _unit(E.mean(0))
    to_centroid = E @ c
    # legit images keep near-perfect peer agreement under the median; the centroid is dragged toward the foreign block
    assert s[:6].min() > 0.9
    assert to_centroid[:6].max() < 0.9
    assert (s[:6].min() - s[6:].max()) > (to_centroid[:6].min() - to_centroid[6:].max())


def test_consensus_small_n_paths():
    E = _unit(np.array([[1.0, 0.0], [0.6, 0.8]]))
    assert np.isnan(consensus_scores(cosine_matrix(E[:1]))).all()
    s = consensus_scores(cosine_matrix(E))
    assert s.shape == (2,) and np.allclose(s, 0.6)


def test_consensus_trimmed_mean_matches_numpy():
    S = np.array([[1, 0.9, 0.8, 0.1, 0.0], [0.9, 1, 0.8, 0.1, 0.0], [0.8, 0.8, 1, 0.1, 0.0], [0.1, 0.1, 0.1, 1, 0.0], [0, 0, 0, 0, 1.0]])
    s = consensus_scores(S, method="trimmed_mean", trim_fraction=0.25, min_peers_for_median=3)
    peers0 = np.sort([0.9, 0.8, 0.1, 0.0])  # trim 25% each side of 4 -> drop 1 each side
    assert np.isclose(s[0], peers0[1:-1].mean())


def test_modified_z_floor_and_constant_input():
    z = modified_z(np.array([1.0, 1.0, 1.0, 1.0, 0.0]), mad_floor=0.01)
    assert np.isfinite(z).all() and z[4] < -3.5 and np.allclose(z[:4], 0)
    assert np.allclose(modified_z(np.ones(5), mad_floor=0.01), 0)
    z = modified_z(np.array([0.5, np.nan, 0.5]), mad_floor=0.01)
    assert np.isnan(z[1]) and np.isfinite(z[[0, 2]]).all()


def test_dual_gate_truth_table():
    z = np.array([-5.0, -5.0, 0.0, 0.0])
    p = np.array([0.9, 0.1, 0.9, 0.1])
    assert dual_gate(z, p, z_thresh=3.5, p_thresh=0.5).tolist() == [True, False, False, False]


def test_support_group_requires_min_size():
    S = np.array([[1, 0.9, 0.85, 0.1], [0.9, 1, 0.8, 0.1], [0.85, 0.8, 1, 0.1], [0.1, 0.1, 0.1, 1.0]])
    g = support_group(S, 0, floor=0.5, min_size=3)
    assert g is not None and sorted(g[0]) == [1, 2] and np.isclose(g[1], 0.875)
    assert support_group(S, 0, floor=0.87, min_size=3) is None  # only one peer above floor
    assert support_group(S, 3, floor=0.5, min_size=3) is None


def test_peer_indices_adds_typical_images_without_duplicates():
    from outlet_audit.verify import peer_indices

    sims = np.array([1.0, 0.9, 0.8, 0.2, 0.1])
    consensus = np.array([0.5, 0.9, 0.3, 0.95, 0.1])
    peers = peer_indices(sims, consensus, k_sim=2, k_typical=2, exclude=0)
    assert peers == [1, 2, 3]  # top-2 by similarity, then the most typical (3); 1 already present
