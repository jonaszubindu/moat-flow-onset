"""Penumbra and moat onsets per azimuthal sector of the tracked spot.

The whole-spot curves average the ring and sum the penumbra over all
azimuths. That is right for a spot that grows its penumbra all round,
but not for one that forms it on one side only (AR13010: southern
sector, 2022-05-16). Here the same quantities are split by azimuth
around the tracked centre:

  penumbral / umbral area : pixels of the tracked spot (tr["masks"]) in
                            the sector
  moat outflow            : mean v_r over annulus & sector pixels, the
                            same contamination mask as the audit
  MMF outflow             : the same on the magnetogram flows

and each sector's curves go through the same onset rule as the whole
spot (moatflow.analysis.onsets), with their own baseline and mature
level. Angles: 0 deg = +x = solar west, 90 deg = +y = north (im_patch
and deprojected cubes are both north-up, west-right).
"""

import numpy as np

from .audit import annulus_geometry, contamination_mask, radial_velocity
from .onsets import (FIRST_MIN_AREA, MOAT_FIRST_FLOOR, holds_above,
                     moat_onset, penumbra_onset)
from .penumbra import segment_frame
from .spottrack import PX_MM

QUADRANTS = {"W": (-45.0, 45.0), "N": (45.0, 135.0),
             "E": (135.0, 225.0), "S": (225.0, 315.0)}
N_AZ = 12                    # 30 deg sectors for the time-azimuth maps
MIN_PX = 15                  # annulus pixels needed for a sector mean
NO_GROWTH_MM2 = FIRST_MIN_AREA / 2   # a quadrant "forms penumbra" above this
NO_MOAT_KMS = MOAT_FIRST_FLOOR


def _in_sector(phi_deg, lo, hi):
    """phi in [lo, hi) with wrap-around (angles in degrees)."""
    return ((phi_deg - lo) % 360.0) < ((hi - lo) % 360.0 or 360.0)


def sector_series(frames, tr, VX, VY, VXm=None, VYm=None, mode="fixed",
                  sectors=QUADRANTS, n_az=N_AZ):
    """Per-epoch, per-sector areas and outflows.

    Returns dict of arrays: a_pen, a_umb, v, v_mmf, frac_bad (n, nsec) for
    `sectors` (in that order, names in "names"), plus az_v, az_pen
    (n, n_az) for the time-azimuth maps (az centres in "az_deg").
    """
    names = list(sectors)
    n, ns = len(frames), len(names)
    out = {k: np.full((n, ns), np.nan)
           for k in ("a_pen", "a_umb", "v", "v_mmf", "frac_bad")}
    az_edges = np.linspace(0, 360, n_az + 1)
    out["az_deg"] = 0.5 * (az_edges[:-1] + az_edges[1:])
    out["az_v"] = np.full((n, n_az), np.nan)
    out["az_pen"] = np.full((n, n_az), np.nan)
    out["names"] = names
    px2 = PX_MM ** 2
    for k in range(n):
        if not tr["valid"][k]:
            continue
        frame = np.asarray(frames[k])
        cy, cx = tr["center"][k]
        rr, phi, ann, _, _ = annulus_geometry(frame.shape, cy, cx,
                                              tr["r_spot_px"][k], mode=mode)
        phid = np.rad2deg(phi) % 360.0
        u, p = segment_frame(frame)
        own = tr["masks"][k]
        pen, umb = p & own, u & own
        bad = contamination_mask(frame, own)
        vr = radial_velocity(VX[k], VY[k], cy, cx, rr)
        vm = (radial_velocity(VXm[k], VYm[k], cy, cx, rr)
              if VXm is not None else None)
        for j, nm in enumerate(names):
            sec = _in_sector(phid, *sectors[nm])
            out["a_pen"][k, j] = (pen & sec).sum() * px2
            out["a_umb"][k, j] = (umb & sec).sum() * px2
            a = ann & sec
            out["frac_bad"][k, j] = (a & bad).sum() / max(a.sum(), 1)
            good = a & ~bad & np.isfinite(vr)
            if good.sum() >= MIN_PX:
                out["v"][k, j] = np.nanmean(vr[good])
            if vm is not None:
                gm = a & ~bad & np.isfinite(vm)
                if gm.sum() >= MIN_PX:
                    out["v_mmf"][k, j] = np.nanmean(vm[gm])
        for j in range(n_az):
            sec = _in_sector(phid, az_edges[j], az_edges[j + 1])
            out["az_pen"][k, j] = (pen & sec).sum() * px2
            good = ann & sec & ~bad & np.isfinite(vr)
            if good.sum() >= MIN_PX:
                out["az_v"][k, j] = np.nanmean(vr[good])
    return out


def sector_onsets(t, ser, valid):
    """Onsets per sector with the whole-spot rule; one dict per sector."""
    res = {}
    for j, nm in enumerate(ser["names"]):
        pen = penumbra_onset(t, ser["a_pen"][:, j], valid)
        moat = moat_onset(t, ser["v"][:, j], valid)
        mmf = (moat_onset(t, ser["v_mmf"][:, j], valid)
               if np.isfinite(ser["v_mmf"][:, j]).any() else None)
        growth = pen["mature"] - pen["zero"]
        forms = bool(np.isfinite(growth) and growth > NO_GROWTH_MM2)
        has_moat = bool(np.isfinite(moat["mature"])
                        and moat["mature"] > NO_MOAT_KMS)
        r = {"pen_baseline_Mm2": pen["zero"], "pen_mature_Mm2": pen["mature"],
             "forms_penumbra": forms,
             "t_pen_h": pen["t"] if forms else np.nan,
             "moat_mature_km_s": moat["mature"], "has_moat": has_moat,
             "t_moat_h": moat["t"] if has_moat else np.nan,
             "t_mmf_h": mmf["t"] if mmf is not None else np.nan,
             "moat_holds": holds_above(t, moat["smooth"], moat["thr"],
                                       moat["t"]),
             "_pen": pen, "_moat": moat, "_mmf": mmf}
        r["lag_moat_h"] = r["t_moat_h"] - r["t_pen_h"]
        r["lag_mmf_h"] = r["t_mmf_h"] - r["t_pen_h"]
        k_on = (np.nanargmin(np.abs(t - r["t_moat_h"]))
                if np.isfinite(r["t_moat_h"]) else None)
        r["contamination_at_moat_onset"] = (
            float(np.nanmax(ser["frac_bad"][max(k_on - 3, 0):k_on + 4, j]))
            if k_on is not None else np.nan)
        res[nm] = r
    return res
