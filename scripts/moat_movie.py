#!/usr/bin/env python
"""Slow, annotated movie showing HOW the moat flow is tracked.

Layout per frame:
  left   45 s continuum, zoomed on the tracked leading spot, with the
         spot segmentation contour, the moat annulus (dashed circles),
         and the hourly FLCT flow arrows
  right  45 s magnetogram (+-150 G) with the same annulus — MMFs are the
         small flux patches streaming outward through it
  bottom moat outflow curve with a time cursor

Usage:
  python scripts/moat_movie.py AR11490 [--stride 8] [--fps 8] [--half 90]

stride 8 at 8 fps = 48 min of solar time per second of video.
"""

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from moatflow.analysis.penumbra import segment_frame
from moatflow.analysis.spottrack import PX_MM, track_spot
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--stride", type=int, default=8,
                    help="use every Nth 45s frame (8 -> 6 min steps)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--half", type=int, default=90,
                    help="half-size of the zoom box [px]")
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    ic, t_iso, _ = load_cube(out / "cube_hmi.Ic_45s_continuum.h5")
    mg, _, _ = load_cube(out / "cube_hmi.M_45s_magnetogram.h5")
    times_s = parse_times(t_iso)

    with h5py.File(out / "flct_hmi.Ic_45s_continuum_w3600s_s5px_k1.h5") as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]

    d = np.load(out / "onset_series_tracked.npz")
    t_curve, v_curve = d["t_h"], d["v_moat"]

    # tracked spot geometry at the flow-window epochs, interpolated to
    # all frame times (curve holds through short tracking dropouts)
    epoch_idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    tr = track_spot([np.asarray(ic[i]) for i in epoch_idx],
                    int(np.argmin(np.abs(t_mid - 82 * 3600))))
    ok = tr["valid"]
    cy = np.interp(times_s, t_mid[ok], tr["center"][ok, 0])
    cx = np.interp(times_s, t_mid[ok], tr["center"][ok, 1])
    rspot = np.interp(times_s, t_mid[ok], tr["r_spot_px"][ok])

    sample = np.asarray(ic[len(ic) // 2])
    ic_kw = dict(cmap="afmhot", vmin=np.percentile(sample, 0.5),
                 vmax=np.percentile(sample, 99.9), origin="lower")
    mg_kw = dict(cmap="gray", vmin=-150, vmax=150, origin="lower")

    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], hspace=0.25)
    axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])
    axC.plot(t_curve, v_curve, "-", color="tab:blue", lw=1.2)
    axC.axhline(0, color="k", lw=0.5)
    axC.axvspan(36, 67, alpha=0.12, color="green")
    axC.set(xlabel="hours since start", ylabel="moat outflow [km/s]")
    cursor = axC.axvline(0, color="crimson", lw=1.2)
    axC.grid(alpha=0.3)

    writer = FFMpegWriter(fps=args.fps, bitrate=4500)
    out_mp4 = out / "quicklook" / "moat_tracking_explained.mp4"
    frames = range(0, len(ic), args.stride)
    print(f"rendering {len(list(frames))} frames ...")

    with writer.saving(fig, str(out_mp4), dpi=110):
        for n in range(0, len(ic), args.stride):
            for ax in (axL, axR):
                ax.clear()
            frame_ic = np.asarray(ic[n])
            frame_mg = np.asarray(mg[n])
            c = (cy[n], cx[n])
            r0 = rspot[n] + 1.0 / PX_MM
            r1 = r0 + 6.0 / PX_MM

            axL.imshow(frame_ic, **ic_kw)
            u, p = segment_frame(frame_ic)
            axL.contour(u | p, levels=[0.5], colors="lime", linewidths=0.7)
            k = int(np.argmin(np.abs(t_mid - times_s[n])))
            s = 6
            Y, X = np.mgrid[0:VX.shape[1]:s, 0:VX.shape[2]:s]
            axL.quiver(X, Y, VX[k][::s, ::s], VY[k][::s, ::s],
                       color="c", scale=10, width=0.0022)
            axL.set_title("continuum + FLCT flows (hourly avg)")

            axR.imshow(frame_mg, **mg_kw)
            axR.set_title("magnetogram $\\pm$150 G — MMFs cross the annulus")

            for ax in (axL, axR):
                for r, ls in ((r0, "--"), (r1, "--")):
                    ax.add_patch(Circle((c[1], c[0]), r, fill=False,
                                        color="crimson", ls=ls, lw=1.1))
                ax.plot(c[1], c[0], "+", color="crimson", ms=10)
                ax.set_xlim(c[1] - args.half, c[1] + args.half)
                ax.set_ylim(c[0] - args.half, c[0] + args.half)
                ax.set_axis_off()

            t_h = times_s[n] / 3600
            cursor.set_xdata([t_h, t_h])
            fig.suptitle(f"{args.event_id} — moat tracking   "
                         f"{t_iso[n][:19]}  (t = {t_h:.1f} h)", y=0.98)
            writer.grab_frame()
    print(f"wrote {out_mp4}")


if __name__ == "__main__":
    main()
