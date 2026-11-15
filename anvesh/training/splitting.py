"""Leakage-aware train/val/test splitting for a dataset with NO camera/sequence identifier.

MIO-TCD localisation frames come from ~8,000 fixed traffic cameras and carry no camera id, so a random
image split would put near-identical frames of the same camera (same background, same lanes, often the
same parked vehicles) on both sides of a split and inflate held-out scores. Mitigation used here:

  1. describe each image by a tiny colour thumbnail (`thumbnail_feature`);
  2. link every pair of images whose thumbnail RMS pixel distance is below a threshold
     (`near_duplicate_groups`, exact search on the GPU/CPU with torch, union-find on the edges);
  3. assign whole connected groups to one split (`assign_groups_to_splits`), so linked images never straddle
     a split;
  4. audit afterwards (`nearest_distance`): how close is each test image to its nearest training image,
     compared with how close a test image is to the rest of the test set.

This is a best-effort proxy for "same camera / same scene", NOT a guarantee: the same camera at night vs day,
or a different crop of the same view, has a large thumbnail distance and can still be split across sets.
The report states this as a limitation.
"""

from __future__ import annotations

import numpy as np

THUMB = 16  # thumbnail side in pixels -> THUMB*THUMB*3 features


def thumbnail_feature(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    small = cv2.resize(image_bgr, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    return small.reshape(-1).astype(np.uint8)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def near_duplicate_groups(features: np.ndarray, threshold: float, chunk: int = 2048, device=None) -> np.ndarray:
    """Connected components of the graph "RMS pixel distance <= threshold". -> group label per row (0..G-1)."""
    import torch

    n, d = features.shape
    device = device or _torch_device()
    x = torch.from_numpy(features.astype(np.float32)).to(device)
    sq = (x * x).sum(1)
    uf = UnionFind(n)
    limit = (threshold ** 2) * d  # squared L2 distance equivalent of the RMS threshold
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d2 = sq[start:stop, None] + sq[None, :] - 2.0 * (x[start:stop] @ x.T)
        # only j > i, so each pair is visited once
        rows = torch.arange(start, stop, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        hit = (d2 <= limit) & (cols > rows)
        ii, jj = torch.nonzero(hit, as_tuple=True)
        for a, b in zip(ii.tolist(), jj.tolist()):
            uf.union(start + a, b)
    roots = np.array([uf.find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def group_size_summary(labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    return {"n_images": int(len(labels)), "n_groups": int(len(sizes)), "largest_group": int(sizes.max()),
            "largest_group_fraction": float(sizes.max() / len(labels)), "singleton_groups": int((sizes == 1).sum()),
            "median_group_size": float(np.median(sizes))}


def assign_groups_to_splits(labels: np.ndarray, fractions: dict, seed: int) -> np.ndarray:
    """Whole groups -> one split each. `fractions` e.g. {"train": .8, "val": .1, "test": .1}. Returns the split name per image.
    Groups are shuffled (seeded) then each goes to the split currently furthest below its image-count target, so a
    huge group cannot starve a split of images by luck of ordering."""
    assert abs(sum(fractions.values()) - 1.0) < 1e-9
    sizes = np.bincount(labels)
    order = np.random.default_rng(seed).permutation(len(sizes))
    total = int(sizes.sum())
    target = {k: v * total for k, v in fractions.items()}
    filled = {k: 0 for k in fractions}
    group_split = {}
    for g in order:
        k = max(fractions, key=lambda s: (target[s] - filled[s]) / target[s])
        group_split[int(g)] = k
        filled[k] += int(sizes[g])
    return np.array([group_split[int(g)] for g in labels])


def nearest_distance(queries: np.ndarray, reference: np.ndarray, chunk: int = 2048, device=None,
                     exclude_self: bool = False) -> np.ndarray:
    """RMS pixel distance from each query row to its nearest reference row (`exclude_self` when query is reference)."""
    import torch

    device = device or _torch_device()
    d = queries.shape[1]
    r = torch.from_numpy(reference.astype(np.float32)).to(device)
    rsq = (r * r).sum(1)
    out = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[start:start + chunk].astype(np.float32)).to(device)
        d2 = (q * q).sum(1)[:, None] + rsq[None, :] - 2.0 * (q @ r.T)
        if exclude_self:
            idx = torch.arange(len(q), device=device)
            d2[idx, idx + start] = float("inf")
        out[start:start + chunk] = (d2.min(1).values.clamp_min(0) / d).sqrt().cpu().numpy()
    return out
