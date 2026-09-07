#!/usr/bin/env python
"""Verification movie: WHAT goes into the moat-outflow curve, frame by frame.

Standard pipeline product. The onset curves are azimuthal means over an
annulus; in a complex group that average can be contaminated by
neighbouring pores/spots and can be dominated by a few sectors. This
movie makes every ingredient visible:

  left    continuum zoom: tracked spot contour (green), moat annulus
          (dashed red), FLCT arrows, and the pixels EXCLUDED from the
          average (other spots/pores + their halo) hatched in red
  top r.  azimuthal breakdown: v_r per 30 deg sector (polar bars,
          outward = positive). Grey sectors are rejected because too
          much of the sector is contaminated. The dashed circle is the
          quoted mean — a healthy moat is a full ring of similar bars,
          not two big bars carrying the average.
  bot r.  radial profile v_r(r) with the annulus band shaded, so you see
          whether the annulus sits on the moat peak.
  bottom  the moat curve with a time cursor, clean vs unmasked.

Usage: python scripts/moat_audit.py AR11210 [--stride 2] [--fps 6]
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
from scipy import ndimage

from moatflow.analysis.penumbra import segment_frame
from moatflow.analysis.spottrack import PX_MM, track_spot
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_times

GAP_MM, WIDTH_MM = 1.0, 6.0
N_SECT = 12
SECT_REJECT = 0.35        # reject a sector if >35 % of it is contaminated


def annulus_geometry(shape, cy, cx, r_spot_px):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    r0 = r_spot_px + GAP_MM / PX_MM
    ann = (rr >= r0) & (rr < r0 + WIDTH_MM / PX_MM)
    phi = np.arctan2(yy - cy, xx - cx)
    return rr, phi, ann, r0


def contamination_mask(frame, own_mask, dilate=3):
    """Pixels belonging to any spot/pore OTHER than the tracked one.

    Dilated, because the granulation next to a dark feature is disturbed
    and its LCT signal is not moat flow either.
    """
    u, p = segment_frame(frame)
    dark = u | p
    others = dark & ~ndimage.binary_dilation(own_mask, iterations=2)
    return ndimage.binary_dilation(others, iterations=dilate)


def sector_profile(vr, ann, phi, bad):
    """(centres, mean v_r, rejected flag) per azimuthal sector."""
    edges = np.linspace(-np.pi, np.pi, N_SECT + 1)
    mid, vals, rejected = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sec = ann & (phi >= a) & (phi < b)
        if sec.sum() == 0:
            continue
        frac_bad = (sec & bad).sum() / sec.sum()
        good = sec & ~bad & np.isfinite(vr)
        mid.append(0.5 * (a + b))
        vals.append(np.nanmean(vr[good]) if good.sum() > 10 else np.nan)
        rejected.append(frac_bad > SECT_REJECT)
    return np.array(mid), np.array(vals), np.array(rejected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth flow window (default: all)")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--half", type=int, default=110, help="zoom half-size [px]")
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    cube, t_iso, _ = load_cube(out / "cube_hmi.Ic_45s_continuum.h5")
    times_s = parse_times(t_iso)
    with h5py.File(out / "flct_hmi.Ic_45s_continuum_w3600s_s5px_k1.h5") as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]
    t_h = t_mid / 3600

    idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    frames = [np.asarray(cube[i]) for i in idx]
    tr = track_spot(frames, int(np.argmin(np.abs(t_h - 0.85 * t_h[-1]))))

    # curves: as-quoted (all annulus pixels) and clean (contamination removed)
    v_raw = np.full(len(frames), np.nan)
    v_clean = np.full(len(frames), np.nan)
    frac_bad = np.full(len(frames), np.nan)
    cache = {}
    for k in range(len(frames)):
        if not tr["valid"][k]:
            continue
        cy, cx = tr["center"][k]
        rr, phi, ann, r0 = annulus_geometry(frames[k].shape, cy, cx,
                                            tr["r_spot_px"][k])
        with np.errstate(invalid="ignore"):
            vr = ((VX[k] * (np.arange(frames[k].shape[1])[None, :] - cx)
                   + VY[k] * (np.arange(frames[k].shape[0])[:, None] - cy))
                  / np.where(rr > 0, rr, np.nan))
        bad = contamination_mask(frames[k], tr["masks"][k])
        cache[k] = (rr, phi, ann, r0, vr, bad)
        ok = ann & np.isfinite(vr)
        v_raw[k] = np.nanmean(vr[ok])
        v_clean[k] = np.nanmean(vr[ok & ~bad])
        frac_bad[k] = (ann & bad).sum() / ann.sum()

    np.savez(out / "moat_audit.npz", t_h=t_h, v_raw=v_raw, v_clean=v_clean,
             frac_contaminated=frac_bad, valid=tr["valid"])
    good = np.isfinite(v_raw) & np.isfinite(v_clean)
    print(f"contamination: median {np.nanmedian(frac_bad)*100:.1f}% of the "
          f"annulus, max {np.nanmax(frac_bad)*100:.1f}%")
    print(f"curve shift when masked: median "
          f"{np.nanmedian(v_clean[good]-v_raw[good])*1000:+.0f} m/s, "
          f"max |shift| {np.nanmax(np.abs(v_clean[good]-v_raw[good]))*1000:.0f} m/s")

    sample = frames[len(frames) // 2]
    ic_kw = dict(cmap="afmhot", origin="lower",
                 vmin=np.percentile(sample, 0.5),
                 vmax=np.percentile(sample, 99.9))

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1],
                          height_ratios=[1.1, 1, 0.55], hspace=0.38,
                          wspace=0.22)
    axIm = fig.add_subplot(gs[0:2, 0])
    axPol = fig.add_subplot(gs[0, 1], projection="polar")
    axRad = fig.add_subplot(gs[1, 1])
    axCur = fig.add_subplot(gs[2, :])

    writer = FFMpegWriter(fps=args.fps, bitrate=4500)
    mp4 = out / "quicklook" / "moat_audit.mp4"
    ks = [k for k in range(0, len(frames), args.stride)]
    print(f"rendering {len(ks)} frames -> {mp4}")

    with writer.saving(fig, str(mp4), dpi=105):
        for k in ks:
            for ax in (axIm, axPol, axRad, axCur):
                ax.clear()
            axCur.plot(t_h, v_raw, "-", color="0.7", lw=1.1,
                       label="annulus mean, unmasked")
            axCur.plot(t_h, v_clean, "-", color="tab:blue", lw=1.3,
                       label="contamination masked (quoted)")
            axCur.axhline(0, color="k", lw=0.5)
            axCur.axvline(t_h[k], color="crimson", lw=1.2)
            axCur.set(xlabel="hours since start", ylabel="v$_r$ [km/s]")
            axCur.legend(fontsize=7.5, loc="upper left")
            axCur.grid(alpha=0.3)

            if k not in cache:
                axIm.text(.5, .5, "tracking lost at this epoch",
                          ha="center", transform=axIm.transAxes)
                axIm.set_axis_off(); axPol.set_axis_off(); axRad.set_axis_off()
                writer.grab_frame()
                continue

            rr, phi, ann, r0, vr, bad = cache[k]
            cy, cx = tr["center"][k]
            axIm.imshow(frames[k], **ic_kw)
            u, p = segment_frame(frames[k])
            axIm.contour(tr["masks"][k], levels=[.5], colors="lime", linewidths=1.0)
            axIm.contourf(bad & ann, levels=[.5, 1.5], colors="none",
                          hatches=["///"], alpha=0)
            axIm.contour(bad & ann, levels=[.5], colors="red", linewidths=0.7)
            s = 7
            Y, X = np.mgrid[0:vr.shape[0]:s, 0:vr.shape[1]:s]
            axIm.quiver(X, Y, VX[k][::s, ::s], VY[k][::s, ::s], color="c",
                        scale=10, width=0.0022)
            for r in (r0, r0 + WIDTH_MM / PX_MM):
                axIm.add_patch(Circle((cx, cy), r, fill=False, color="crimson",
                                      ls="--", lw=1.1))
            axIm.plot(cx, cy, "+", color="crimson", ms=11)
            axIm.set_xlim(cx - args.half, cx + args.half)
            axIm.set_ylim(cy - args.half, cy + args.half)
            axIm.set_axis_off()
            axIm.set_title("green = tracked spot | red = excluded (other "
                           "spots) | dashed = moat annulus", fontsize=9)

            mid, vals, rej = sector_profile(vr, ann, phi, bad)
            width = 2 * np.pi / len(mid)
            colors = ["0.75" if r else "tab:blue" for r in rej]
            axPol.bar(mid, np.nan_to_num(vals), width=width * 0.9,
                      color=colors, alpha=0.85)
            m = np.nanmean(vals[~rej]) if (~rej).any() else np.nan
            th = np.linspace(-np.pi, np.pi, 100)
            axPol.plot(th, np.full_like(th, m), "k--", lw=1.0)
            axPol.set_ylim(min(-0.15, np.nanmin(vals) - .05),
                           max(0.55, np.nanmax(vals) + .05))
            axPol.set_title(f"v$_r$ per sector  (mean {m:.2f} km/s)\n"
                            "grey = rejected, dashed = mean", fontsize=8.5)
            axPol.tick_params(labelsize=6)

            edges = np.arange(0, args.half, 3.0)
            rmid, rprof = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = (rr >= a) & (rr < b) & ~bad & np.isfinite(vr)
                if sel.sum() > 20:
                    rmid.append(0.5 * (a + b) * PX_MM)
                    rprof.append(np.nanmean(vr[sel]))
            axRad.axhline(0, color="k", lw=0.5)
            axRad.plot(rmid, rprof, "o-", ms=3, color="tab:blue")
            axRad.axvspan(r0 * PX_MM, (r0 + WIDTH_MM / PX_MM) * PX_MM,
                          alpha=0.15, color="crimson")
            axRad.set(xlabel="r from spot centre [Mm]", ylabel="v$_r$ [km/s]",
                      ylim=(-0.3, 0.7))
            axRad.set_title("radial profile; shaded = annulus", fontsize=8.5)
            axRad.grid(alpha=0.3)

            fig.suptitle(f"{args.event_id} moat audit — t = {t_h[k]:.1f} h  "
                         f"({t_iso[idx[k]][:16]})   "
                         f"annulus contaminated: {frac_bad[k]*100:.0f}%",
                         y=0.98, fontsize=11)
            writer.grab_frame()
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
