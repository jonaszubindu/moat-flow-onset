#!/usr/bin/env python
"""Physical verifications that FLCT tracks real average flows.

Two standard literature checks, using only data already on disk:

1. Divergence map (`<event>_flct_divergence.png`)
   The time-averaged divergence of the LCT flow field must show the
   supergranular cell pattern (~20-35 Mm cells, ~+-4e-5 /s) in quiet
   regions, and the moat as a divergence structure around the mature
   spot. Random tracking noise shows no such organization.

2. Doppler cross-check (`<event>_flct_doppler_check.png`)
   Away from disk center, horizontal outflows have a line-of-sight
   component: redshift on the limb side of the spot, blueshift on the
   disk-center side. The azimuthal pattern of the (background-removed,
   time-averaged) Dopplergram around the spot is compared with the
   LOS-projection of the FLCT flow field. Agreement in phase and shape
   verifies the tracked flows against an independent observable.

Usage: python scripts/verify_flct.py AR11490 --t-quiet 82 --t-doppler 88
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from moatflow.analysis.spottrack import PX_MM, track_spot
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes, parse_times

KM_PER_PX = 362.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--t-quiet", type=float, default=82.0,
                    help="epoch [h] for the divergence map (mature spot)")
    ap.add_argument("--t-doppler", type=float, default=88.0,
                    help="epoch [h] for the Doppler check — pick a time "
                         "well away from disk-center passage")
    ap.add_argument("--lon0", type=float, required=True,
                    help="Stonyhurst lon [deg] of the AR at t=0 of the "
                         "window (from events.yaml resolution)")
    ap.add_argument("--lat", type=float, required=True,
                    help="AR latitude [deg]")
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    ic, t_iso, _ = load_cube(out / "cube_hmi.Ic_45s_continuum.h5")
    times_s = parse_times(t_iso)
    with h5py.File(out / "flct_hmi.Ic_45s_continuum_w3600s_s5px_k1.h5") as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]

    # tracked spot geometry
    epoch_idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    tr = track_spot([np.asarray(ic[i]) for i in epoch_idx],
                    int(np.argmin(np.abs(t_mid - args.t_quiet * 3600))))

    # ---- 1. divergence map -------------------------------------------
    k = int(np.argmin(np.abs(t_mid - args.t_quiet * 3600)))
    vx = VX[max(k-1, 0):k+2].mean(0)
    vy = VY[max(k-1, 0):k+2].mean(0)
    div = (np.gradient(vx, KM_PER_PX, axis=1)
           + np.gradient(vy, KM_PER_PX, axis=0))
    div_s = ndimage.gaussian_filter(div, 8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    frame = np.asarray(ic[epoch_idx[k]])
    axes[0].imshow(frame, cmap="afmhot", origin="lower",
                   vmin=np.percentile(frame, 0.5),
                   vmax=np.percentile(frame, 99.9))
    axes[0].set_title(f"continuum, t={t_mid[k]/3600:.0f} h")
    im = axes[1].imshow(div_s * 1e5, cmap="RdBu_r", origin="lower",
                        vmin=-1.5, vmax=1.5)
    axes[1].set_title("FLCT divergence [$10^{-5}$ s$^{-1}$], 3 h avg\n"
                      "supergranular cells + moat ring = real flows")
    plt.colorbar(im, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_axis_off()
    cyx = tr["center"][k]
    for ax in axes:
        ax.plot(cyx[1], cyx[0], "g+", ms=12)
    fig.tight_layout()
    f1 = out / "quicklook" / "flct_divergence.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    # ---- 2. Doppler cross-check --------------------------------------
    vc, _, _ = load_cube(out / "cube_hmi.V_45s_Dopplergram.h5")
    k = int(np.argmin(np.abs(t_mid - args.t_doppler * 3600)))
    i0 = epoch_idx[k]
    n_avg = 160                        # 2 h of 45 s frames
    dop = np.mean([np.asarray(vc[i]) for i in
                   range(max(0, i0 - n_avg // 2),
                         min(len(vc), i0 + n_avg // 2), 4)], axis=0)
    # remove large-scale background (rotation gradient, convective
    # blueshift, meridional terms): subtract a 60-px gaussian smooth
    dop_res = dop - ndimage.gaussian_filter(dop, 60)

    vx = VX[max(k-1, 0):k+2].mean(0)
    vy = VY[max(k-1, 0):k+2].mean(0)

    # AR heliographic position at this epoch (synodic rotation)
    lon = args.lon0 + 13.2 / 24.0 * (t_mid[k] / 3600)
    l, b = np.deg2rad(lon), np.deg2rad(args.lat)
    # LOS (away from observer) component of the horizontal flow at
    # heliographic (l, b), B0 neglected: westward motion recedes as
    # cos(b) sin(l); northward motion recedes as sin(b) cos(l)
    # (negative in the southern hemisphere: northward = toward observer).
    v_los_pred_kms = (vx * np.cos(b) * np.sin(l)
                      + vy * np.sin(b) * np.cos(l))

    cy, cx = tr["center"][k]
    yy, xx = np.mgrid[0:dop.shape[0], 0:dop.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    r0 = tr["r_spot_px"][k] + 1.0 / PX_MM
    ann = (rr >= r0) & (rr < r0 + 6.0 / PX_MM)
    phi = np.arctan2(yy - cy, xx - cx)

    bins = np.linspace(-np.pi, np.pi, 19)
    phic = 0.5 * (bins[:-1] + bins[1:])
    prof_d, prof_p = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = ann & (phi >= a) & (phi < b)
        prof_d.append(np.nanmean(dop_res[sel]) / 1e3)   # m/s -> km/s
        prof_p.append(np.nanmean(v_los_pred_kms[sel]))
    prof_d, prof_p = np.array(prof_d), np.array(prof_p)
    cc = np.corrcoef(prof_d, prof_p)[0, 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.rad2deg(phic), prof_d, "o-", label="Dopplergram (2 h avg,"
            " background-removed)")
    ax.plot(np.rad2deg(phic), prof_p, "s-", label="FLCT flows projected"
            " onto LOS")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="azimuth around spot [deg]  (0 = solar W)",
           ylabel="LOS velocity in moat annulus [km/s]",
           title=f"{args.event_id} Doppler cross-check, t={t_mid[k]/3600:.0f} h"
                 f" (AR at lon {lon:.0f} deg) — r = {cc:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = out / "quicklook" / "flct_doppler_check.png"
    fig.savefig(f2, dpi=140)
    print(f"wrote {f1}\nwrote {f2}\nazimuthal correlation r = {cc:.2f}")


if __name__ == "__main__":
    main()
