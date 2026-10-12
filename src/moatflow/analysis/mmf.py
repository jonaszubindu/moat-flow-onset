"""Moving magnetic features (MMFs): detection, tracking and an onset time.

The FLCT magnetogram flow (v_mmf) measures an averaged outflow of flux
patterns. Here MMFs are followed as individual features, which is what
high-resolution observations show at penumbra formation: small
opposite-polarity patches leaving the spot.

Per magnetogram frame (every `stride`-th 45 s frame):
  * noise: robust sigma of the quiet pixels (|B| < 50 G);
  * features: connected regions with |B| > T_LOW*sigma that contain a
    pixel above T_HIGH*sigma and cover >= MIN_PX pixels, per polarity;
  * the tracked spot (its hourly mask, moved with the interpolated
    centre, widened by EXCL_MM) and everything farther than R_MAX_MM from
    its edge are left out.
Features are linked between consecutive frames by pixel overlap, else by
the nearest centroid within LINK_PX; splits and merges continue the
branch with the largest overlap.

A track is an outward MMF if it lives >= MIN_LIFE_MIN, is born within
BIRTH_MM of the spot edge, and moves outward (distance from the edge,
which follows the moving, growing spot) by >= MIN_DISP_MM at a fitted
speed >= MIN_V_KMS, with no single step faster than MAX_STEP_KMS (link
jumps between features, or jumps of the equal-area spot edge while pores
fragment). Polarity is relative to the umbra: 'opposite' or
'same' (Shine & Title 2001: type III and type II; bipolar pairs show up
in both). The onset uses the birth rate of outward opposite-polarity
MMFs per hour and the same rule as the other curves (onsets.py).
"""

import numpy as np
from scipy import ndimage

from .onsets import (BASELINE_H, FIRST_SIGMA, FRACTION, MATURE_PCT,
                     first_persistent, holds_above, smooth)
from .spottrack import PX_MM

T_LOW, T_HIGH = 3.0, 5.0
MIN_PX = 4
EXCL_MM = 1.0
R_MAX_MM = 25.0
LINK_PX = 2.0
MIN_LIFE_MIN = 15.0
BIRTH_MM = 8.0
MIN_DISP_MM = 1.5
MIN_V_KMS = 0.2
MAX_STEP_KMS = 5.0             # faster steps are link jumps, not motion
# first appearance: at least one extra MMF per 3 h smoothing window, held
# for 3 h -- "from when on do they keep coming"; a per-hour minimum is
# out of reach for sparse events (AR11184: ~1-2 MMFs/h)
MIN_COUNT_FIRST = 1.0 / 3.0
PX_AREA_CM2 = (PX_MM * 1e8) ** 2


def frame_noise(B):
    q = np.abs(B) < 50
    v = B[q] if q.sum() > 1000 else B.ravel()
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def detect(B, excl, cy, cx, r_spot_px, sigma):
    """Features of one frame: (labels_pos, labels_neg, list of dicts)."""
    yy, xx = np.mgrid[0:B.shape[0], 0:B.shape[1]]
    d_edge = np.hypot(xx - cx, yy - cy) * PX_MM - r_spot_px * PX_MM
    roi = ~excl & (d_edge <= R_MAX_MM)
    feats, labs = [], []
    for sgn in (1, -1):
        sb = sgn * B
        low = (sb > T_LOW * sigma) & roi
        lab, n = ndimage.label(low)
        if n:
            idx = np.arange(1, n + 1)
            size = ndimage.sum_labels(low, lab, idx)
            peak = ndimage.maximum(sb, lab, idx)
            keep = (size >= MIN_PX) & (peak >= T_HIGH * sigma)
            remap = np.zeros(n + 1, int)
            remap[1:][keep] = np.arange(1, keep.sum() + 1)
            lab = remap[lab]
            kept = idx[keep]
            flux = ndimage.sum_labels(sb, lab, np.arange(1, keep.sum() + 1))
            cyx = ndimage.center_of_mass(np.where(lab > 0, sb, 0), lab,
                                         np.arange(1, keep.sum() + 1))
            for j, (fy, fx) in enumerate(cyx):
                feats.append({"pol": sgn, "lab": j + 1, "y": fy, "x": fx,
                              "flux_Mx": float(flux[j]) * PX_AREA_CM2,
                              "npx": int(size[kept[j] - 1])})
        labs.append(lab)
    return labs[0], labs[1], feats


def link(prev, cur):
    """Map each current feature to a previous one (index) or -1.

    prev, cur: (labels_pos, labels_neg, feats)."""
    parent = [-1] * len(cur[2])
    best = [0] * len(cur[2])
    for pi, sgn in ((0, 1), (1, -1)):
        lp, lc = prev[pi], cur[pi]
        both = (lp > 0) & (lc > 0)
        if both.any():
            key = lp[both].astype(np.int64) * 1_000_000 + lc[both]
            u, cnt = np.unique(key, return_counts=True)
            pmap = {f["lab"]: i for i, f in enumerate(prev[2]) if f["pol"] == sgn}
            cmap = {f["lab"]: i for i, f in enumerate(cur[2]) if f["pol"] == sgn}
            for k, c in zip(u, cnt):
                ip, ic = pmap[int(k // 1_000_000)], cmap[int(k % 1_000_000)]
                if c > best[ic]:
                    best[ic], parent[ic] = c, ip
    # no overlap: nearest previous centroid of the same polarity
    for ic, f in enumerate(cur[2]):
        if parent[ic] >= 0:
            continue
        cand = [(np.hypot(f["x"] - g["x"], f["y"] - g["y"]), ip)
                for ip, g in enumerate(prev[2]) if g["pol"] == f["pol"]]
        if cand:
            d, ip = min(cand)
            if d <= LINK_PX:
                parent[ic] = ip
    return parent, best


def build_tracks(frames):
    """frames: list of (t_s, (lab_pos, lab_neg, feats), cy, cx, r_spot_px).
    Returns tracks: list of dicts with arrays t, x, y, d_edge_mm, flux."""
    tracks, open_of_prev = [], {}
    prev = None
    for n, (t, det, cy, cx, rsp) in enumerate(frames):
        feats = det[2]
        open_now = {}
        if prev is not None:
            parent, ov = link(prev[1], det)
            # a parent continues into its child with the largest overlap
            claim = {}
            for ic, ip in enumerate(parent):
                if ip >= 0 and (ip not in claim or ov[ic] > ov[claim[ip]]):
                    claim[ip] = ic
            cont = {ic: ip for ip, ic in claim.items()}
        else:
            cont = {}
        for ic, f in enumerate(feats):
            d = np.hypot(f["x"] - cx, f["y"] - cy) * PX_MM - rsp * PX_MM
            rec = (t, f["x"], f["y"], d, f["flux_Mx"])
            if ic in cont and cont[ic] in open_of_prev:
                ti = open_of_prev[cont[ic]]
            else:
                ti = len(tracks)
                tracks.append({"pol": f["pol"], "rows": []})
            tracks[ti]["rows"].append(rec)
            open_now[ic] = ti
        open_of_prev, prev = open_now, (t, det)
    out = []
    for tr in tracks:
        a = np.array(tr["rows"])
        out.append({"pol": tr["pol"], "t": a[:, 0], "x": a[:, 1], "y": a[:, 2],
                    "d_edge_mm": a[:, 3], "flux_Mx": a[:, 4]})
    return out


def classify(tracks, spot_pol):
    """Per track: lifetime, birth distance, radial speed, net displacement,
    relative polarity, outward flag."""
    rows = []
    for tr in tracks:
        t, d = tr["t"], tr["d_edge_mm"]
        life = (t[-1] - t[0]) / 60.0
        v = (np.polyfit(t, d, 1)[0] * 1e3) if len(t) >= 3 else np.nan  # Mm/s -> km/s
        disp = d[-1] - d[0]
        step = (np.max(np.abs(np.diff(d)) / np.diff(t)) * 1e3) if len(t) > 1 else 0.0
        outward = (life >= MIN_LIFE_MIN and d[0] <= BIRTH_MM
                   and disp >= MIN_DISP_MM and np.isfinite(v) and v >= MIN_V_KMS
                   and step <= MAX_STEP_KMS)
        rows.append({"t_birth": t[0], "life_min": life, "d_birth_mm": d[0],
                     "v_kms": v, "disp_mm": disp,
                     "rel_pol": "opposite" if tr["pol"] != spot_pol else "same",
                     "flux_Mx": float(np.median(tr["flux_Mx"])),
                     "outward": bool(outward)})
    return rows


def hourly_rate(rows, t_h, rel_pol, covered):
    """Births per hour of outward MMFs of one relative polarity, on the
    hourly epochs t_h (bin = epoch +- 0.5 h); NaN where not covered."""
    tb = np.array([r["t_birth"] / 3600 for r in rows
                   if r["outward"] and r["rel_pol"] == rel_pol])
    rate = np.array([np.sum(np.abs(tb - t) < 0.5) for t in t_h], dtype=float)
    return np.where(covered, rate, np.nan)


def mmf_onset(t_h, rate, fraction=FRACTION):
    """Fraction-of-mature onset and first appearance of an MMF rate curve,
    with the same baseline / smoothing / persistence as the other curves."""
    ys = smooth(rate)
    good = np.isfinite(ys)
    if good.sum() < 5:
        return {"t": np.nan, "t_first": np.nan, "zero": np.nan,
                "mature": np.nan, "thr": np.nan, "holds": np.nan, "smooth": ys}
    t0 = float(t_h[good][0])
    base = rate[np.isfinite(rate) & (t_h < t0 + BASELINE_H)]
    zero = float(np.median(base))
    sig = float(1.4826 * np.median(np.abs(base - zero))) / np.sqrt(3.0)
    mature = float(np.nanpercentile(ys[good], MATURE_PCT))
    thr = zero + fraction * (mature - zero)
    t_on = first_persistent(t_h, ys, thr)
    t_first = first_persistent(t_h, ys, zero + max(FIRST_SIGMA * sig, MIN_COUNT_FIRST))
    return {"t": t_on, "t_first": t_first, "zero": zero, "mature": mature,
            "thr": thr, "holds": holds_above(t_h, ys, thr, t_on),
            "t_first_valid": t0, "smooth": ys}
