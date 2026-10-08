"""Track one spot identity through a cube.

The per-frame segmentation is noisy: the leading spot can vanish for a
frame (bad normalization), or merge with neighboring pores through a
transient dark bridge, which makes independently-selected components
jump in area and position. Tracking fixes the identity: seed the spot
once at an epoch where it is unambiguous, then propagate the mask
frame-to-frame by overlap, in both time directions.

Merge guard: penumbral pixels are only counted within `r_max_mm` of the
tracked center, so a transient bridge to a neighboring pore group does
not double the area for one epoch.
"""

import numpy as np
from scipy import ndimage

from .penumbra import segment_frame

PX_MM = 0.362


def _components(frame, **seg_kwargs):
    u, p = segment_frame(frame, **seg_kwargs)
    lab, n = ndimage.label(u | p)
    return u, p, lab, n


def _pick_seed(frame, prefer_west=True, **seg_kwargs):
    """Component of the largest (or westernmost of the two largest) umbra."""
    u, p, lab, n = _components(frame, **seg_kwargs)
    ulab, un = ndimage.label(u)
    if un == 0:
        raise ValueError("no umbra in seed frame")
    sizes = ndimage.sum_labels(u, ulab, index=np.arange(1, un + 1))
    cand = (np.argsort(sizes)[-2:] + 1) if (prefer_west and un >= 2) \
        else np.array([int(np.argmax(sizes)) + 1])
    xc = [ndimage.center_of_mass(ulab == b)[1] for b in cand]
    useed = ulab == cand[int(np.argmax(xc))]
    cy, cx = ndimage.center_of_mass(useed)
    return lab == lab[int(round(cy)), int(round(cx))]


def track_spot(cube, idx_seed: int, r_max_mm: float = 14.0,
               seg_kwargs: dict | None = None):
    """Track the seeded spot over all frames of `cube`.

    Returns dict of arrays over frames: center (n,2), area_umbra,
    area_penumbra [Mm^2], r_spot_px, valid (False where the mask was
    carried over from the previous frame because segmentation lost it),
    plus seed_idx -- the frame actually used as seed, which differs from
    idx_seed when that frame has no umbra (seed_fallback = True).
    """
    seg_kwargs = seg_kwargs or {}
    n = len(cube)
    out = {"center": np.full((n, 2), np.nan),
           "area_umbra": np.full(n, np.nan),
           "area_penumbra": np.full(n, np.nan),
           "r_spot_px": np.full(n, np.nan),
           "valid": np.zeros(n, bool)}
    masks = [None] * n
    try:
        seed = _pick_seed(np.asarray(cube[idx_seed]), **seg_kwargs)
        fallback = False
    except ValueError:
        # No umbra at the requested epoch: the spot has not formed yet or
        # has already decayed (AR11242 decays inside its window). Seed
        # where the umbra is largest instead -- the least ambiguous epoch.
        umbra_px = [segment_frame(np.asarray(f), **seg_kwargs)[0].sum()
                    for f in cube]
        best = int(np.argmax(umbra_px))
        if umbra_px[best] == 0:
            raise ValueError("no umbra in any frame: nothing to track")
        print(f"  [track_spot] no umbra at the requested seed (frame "
              f"{idx_seed}); seeding at frame {best}, where the umbra is "
              "largest")
        idx_seed, fallback = best, True
        seed = _pick_seed(np.asarray(cube[idx_seed]), **seg_kwargs)
    out["seed_idx"] = idx_seed
    out["seed_fallback"] = fallback

    def step(i, prev_mask):
        u, p, lab, ncomp = _components(np.asarray(cube[i]), **seg_kwargs)
        if ncomp == 0:
            return None
        overlap = ndimage.sum_labels(prev_mask, lab,
                                     index=np.arange(1, ncomp + 1))
        if overlap.max() == 0:
            return None
        mask = lab == int(np.argmax(overlap)) + 1
        cy, cx = ndimage.center_of_mass(mask)
        yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        near = np.hypot(xx - cx, yy - cy) <= r_max_mm / PX_MM
        out["center"][i] = cy, cx
        out["area_umbra"][i] = (u & mask & near).sum() * PX_MM ** 2
        out["area_penumbra"][i] = (p & mask & near).sum() * PX_MM ** 2
        out["r_spot_px"][i] = np.sqrt((mask & near).sum() / np.pi)
        out["valid"][i] = True
        return mask & near

    for rng in (range(idx_seed, n), range(idx_seed - 1, -1, -1)):
        prev = seed
        for i in rng:
            m = step(i, prev)
            if m is None:          # lost: carry mask, mark invalid
                masks[i] = prev
                out["center"][i] = out["center"][i - 1] \
                    if rng.step == 1 and i > 0 else out["center"][i]
            else:
                masks[i] = m
                prev = m
    out["masks"] = masks
    return out
