import numpy as np

from anvesh.training.splitting import (
    assign_groups_to_splits, group_size_summary, near_duplicate_groups, nearest_distance)


def _features(rng):
    base_a = rng.integers(0, 255, 48)
    base_b = rng.integers(0, 255, 48)
    rows = [base_a, np.clip(base_a + rng.integers(-3, 4, 48), 0, 255),   # camera A, two near-identical frames
            base_b, np.clip(base_b + rng.integers(-3, 4, 48), 0, 255),   # camera B
            rng.integers(0, 255, 48)]                                     # unrelated
    return np.array(rows, dtype=np.uint8)


def test_near_duplicates_are_grouped_and_unrelated_images_are_not():
    labels = near_duplicate_groups(_features(np.random.default_rng(0)), threshold=10.0, device="cpu")
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert len({labels[0], labels[2], labels[4]}) == 3
    s = group_size_summary(labels)
    assert s["n_groups"] == 3 and s["largest_group"] == 2 and s["singleton_groups"] == 1


def test_groups_never_straddle_splits_and_assignment_is_seeded_and_close_to_targets():
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(300), rng.integers(1, 9, 300))
    fr = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = assign_groups_to_splits(labels, fr, seed=7)
    assert (a == assign_groups_to_splits(labels, fr, seed=7)).all()
    assert not (a == assign_groups_to_splits(labels, fr, seed=8)).all()
    for g in np.unique(labels):
        assert len(set(a[labels == g])) == 1
    for k, v in fr.items():
        assert abs((a == k).mean() - v) < 0.03


def test_nearest_distance_to_train_detects_a_leaked_copy():
    rng = np.random.default_rng(2)
    train = rng.integers(0, 255, (50, 48)).astype(np.uint8)
    leaked = np.clip(train[:1].astype(int) + 2, 0, 255).astype(np.uint8)
    fresh = rng.integers(0, 255, (1, 48)).astype(np.uint8)
    d = nearest_distance(np.vstack([leaked, fresh]), train, device="cpu")
    assert d[0] < 3 < 30 < d[1]
    self_d = nearest_distance(train, train, device="cpu", exclude_self=True)
    assert (self_d > 0).all()
