"""Radial structure of the flows around the spot vs time (tests 1 and 2).

Question: is the moat already there before the penumbra, hidden -- below
the surface (granules only feel the top 1-2 Mm), or at the surface but
behind an inflow ring at the pore edge (Rempel 2015: a naked spot has a
1-2 km/s inflow at its boundary and a ~0.5 km/s outflow peaking ~4 Mm
out; Deng et al. 2007 saw such a ring around a spot that lost its
penumbra)? Then the moat would "appear" when the penumbra replaces the
inflow ring, not when an outflow is created.

Test 1 -- granulation: azimuthally averaged FLCT v_r against distance from
the spot edge (equal-area radius), per hourly epoch; other spots/pores
masked as in the audit; the uniform background flow cancels in the
azimuthal mean.
Test 2 -- magnetic elements: radial drift of long-lived tracked flux
features (mmf_tracks.npz from mmf_onset.py), each speed over a ~30 min
baseline (3 min steps are dominated by position jitter), on the same
time x distance grid. Network elements reach deeper than granules, so an
outflow below the granulation could show up here first.
"""

import numpy as np

from .audit import contamination_mask, radial_velocity
from .spottrack import PX_MM

D_EDGES = np.arange(-4.0, 25.0 + 1e-6, 1.0)       # distance from the edge [Mm]
D_MID = 0.5 * (D_EDGES[:-1] + D_EDGES[1:])
MIN_PX_BIN = 20
MIN_LIFE_MAG_MIN = 60.0
BASELINE_S = (25 * 60, 35 * 60)
WINDOWS = {"pre": (-14.0, -2.0), "onset": (-2.0, 4.0), "post": (6.0, 18.0)}
NEAR_MM, OUTER_MM = (0.0, 3.0), (5.0, 12.0)


def granular_map(frames, tr, VX, VY):
    """(n_epochs, n_bins) azimuthal mean v_r [km/s] vs distance from edge."""
    n = len(frames)
    out = np.full((n, len(D_MID)), np.nan)
    for k in range(n):
        if not tr["valid"][k]:
            continue
        frame = np.asarray(frames[k])
        cy, cx = tr["center"][k]
        yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
        rr = np.hypot(xx - cx, yy - cy)
        d = (rr - tr["r_spot_px"][k]) * PX_MM
        vr = radial_velocity(VX[k], VY[k], cy, cx, rr)
        good = np.isfinite(vr) & ~contamination_mask(frame, tr["masks"][k]) \
            & (d >= D_EDGES[0]) & (d < D_EDGES[-1])
        b = np.digitize(d[good], D_EDGES) - 1
        cnt = np.bincount(b, minlength=len(D_MID))
        s = np.bincount(b, weights=vr[good], minlength=len(D_MID))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(cnt >= MIN_PX_BIN, s / cnt, np.nan)
    return out


def magnetic_drift_map(npz, t_h, exclude_outward_mmf=False):
    """Mean radial drift [km/s] of long-lived features per (hourly epoch,
    distance bin), the counts, and the sums (for count-weighted averages
    over windows -- cells hold few features). npz: mmf_tracks.npz."""
    tid, t, dd = npz["pos_track"], npz["pos_t"], npz["pos_d"]
    life = npz["life_min"]
    outward = npz["outward"].astype(bool)
    keep = life >= MIN_LIFE_MAG_MIN
    if exclude_outward_mmf:
        keep &= ~outward
    order = np.lexsort((t, tid))
    tid, t, dd = tid[order], t[order], dd[order]
    sums = np.zeros((len(t_h), len(D_MID)))
    cnts = np.zeros_like(sums)
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(tid)]
    for a, b in zip(starts, ends):
        if not keep[int(tid[a])] or b - a < 3:
            continue
        tt, d = t[a:b], dd[a:b]
        j = np.searchsorted(tt, tt + BASELINE_S[0])
        ok = j < len(tt)
        i = np.flatnonzero(ok)
        j = j[ok]
        dt = tt[j] - tt[i]
        sel = dt <= BASELINE_S[1]
        i, j, dt = i[sel], j[sel], dt[sel]
        if not len(i):
            continue
        v = (d[j] - d[i]) / dt * 1e3                     # Mm/s -> km/s
        tm = 0.5 * (tt[i] + tt[j]) / 3600
        dm = 0.5 * (d[i] + d[j])
        ki = np.clip(np.round(np.interp(tm, t_h, np.arange(len(t_h)))).astype(int),
                     0, len(t_h) - 1)
        bi = np.digitize(dm, D_EDGES) - 1
        g = (bi >= 0) & (bi < len(D_MID)) & (np.abs(t_h[ki] - tm) <= 0.5)
        np.add.at(sums, (ki[g], bi[g]), v[g])
        np.add.at(cnts, (ki[g], bi[g]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnts >= 3, sums / cnts, np.nan), cnts, sums


def v1_mmf_mask(npz):
    """Tracks meeting the version-1 MMF criteria, recomputed from the saved
    per-track values (works on version-2 files; the link-jump filter is not
    re-checked)."""
    from . import mmf
    return ((npz["life_min"] >= mmf.MIN_LIFE_MIN) & (npz["d_birth_mm"] <= mmf.BIRTH_MM)
            & (npz["disp_mm"] >= mmf.MIN_DISP_MM) & (npz["v_kms"] >= mmf.MIN_V_KMS))


def magnetic_drift_map_fit(npz, t_h, min_samples=3, exclude_v1_mmf=False):
    """Version-2 style test 2: every track with >= min_samples positions
    contributes ONE drift value (its fitted radial speed, npz v_kms) at its
    middle time and middle distance -- short tracks cannot use 30 min
    baselines; noise averages out over many tracks. Returns (mean, counts,
    sums) on the hourly epochs x distance bins."""
    tid, t, dd = npz["pos_track"], npz["pos_t"], npz["pos_d"]
    v = npz["v_kms"]
    skip = v1_mmf_mask(npz) if exclude_v1_mmf else np.zeros(len(v), bool)
    n = np.bincount(tid, minlength=len(v))
    tm = np.bincount(tid, weights=t, minlength=len(v)) / np.maximum(n, 1) / 3600
    dm = np.bincount(tid, weights=dd, minlength=len(v)) / np.maximum(n, 1)
    use = (n >= min_samples) & np.isfinite(v) & ~skip
    ki = np.clip(np.round(np.interp(tm, t_h, np.arange(len(t_h)))).astype(int), 0, len(t_h) - 1)
    bi = np.digitize(dm, D_EDGES) - 1
    g = use & (bi >= 0) & (bi < len(D_MID)) & (np.abs(t_h[ki] - tm) <= 0.5)
    sums = np.zeros((len(t_h), len(D_MID)))
    cnts = np.zeros_like(sums)
    np.add.at(sums, (ki[g], bi[g]), v[g])
    np.add.at(cnts, (ki[g], bi[g]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnts >= 3, sums / cnts, np.nan), cnts, sums


def window_weighted(t_rel, sums, cnts, windows=WINDOWS, min_count=10):
    """Count-weighted mean drift profile in each window (all measurements
    in the window and distance bin pooled)."""
    out = {}
    for name, (a, b) in windows.items():
        sel = (t_rel >= a) & (t_rel < b)
        s_, c_ = sums[sel].sum(axis=0), cnts[sel].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(c_ >= min_count, s_ / c_, np.nan)
    return out


def band_weighted(t_rel, sums, cnts, window, band, min_count=10):
    """Count-weighted mean drift over a time window and a distance band."""
    ts = (t_rel >= window[0]) & (t_rel < window[1])
    ds = (D_MID >= band[0]) & (D_MID < band[1])
    c_ = cnts[np.ix_(ts, ds)].sum()
    return float(sums[np.ix_(ts, ds)].sum() / c_) if c_ >= min_count else np.nan


def window_profiles(t_rel, m, windows=WINDOWS, min_epochs=3):
    """Mean profile in each window of t_rel (hours from penumbra onset)."""
    out = {}
    for name, (a, b) in windows.items():
        sel = (t_rel >= a) & (t_rel < b)
        rows = m[sel]
        okrows = np.isfinite(rows).any(axis=1)
        if okrows.sum() >= min_epochs:
            with np.errstate(all="ignore"):
                out[name] = np.nanmean(rows[okrows], axis=0)
        else:
            out[name] = np.full(m.shape[1], np.nan)
    return out


def band_mean(prof, band):
    sel = (D_MID >= band[0]) & (D_MID < band[1])
    with np.errstate(all="ignore"):
        return float(np.nanmean(prof[sel])) if np.isfinite(prof[sel]).any() else np.nan


def stack(maps, t_rels, grid=np.arange(-30.0, 40.0 + 1e-6, 1.0)):
    """Average maps (n_epochs_i, n_bins) over events on a common grid of
    hours from penumbra onset; returns (grid, mean, count)."""
    acc = np.zeros((len(grid), len(D_MID)))
    cnt = np.zeros_like(acc)
    for m, tr in zip(maps, t_rels):
        for j in range(m.shape[1]):
            col = m[:, j]
            g = np.isfinite(col)
            if g.sum() < 2:
                continue
            v = np.interp(grid, tr[g], col[g], left=np.nan, right=np.nan)
            # no interpolation across gaps longer than 2 h
            near = np.min(np.abs(grid[:, None] - tr[g][None, :]), axis=1) <= 1.0
            v = np.where(near, v, np.nan)
            ok = np.isfinite(v)
            acc[ok, j] += v[ok]
            cnt[ok, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return grid, np.where(cnt > 0, acc / cnt, np.nan), cnt
