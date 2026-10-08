#!/usr/bin/env python
"""Slow, annotated movie showing HOW the moat flow is tracked.

Layout per frame:
  left   45 s continuum, zoomed on the tracked leading spot, with the
         spot segmentation contour, the moat annulus (dashed circles),
         and the hourly FLCT flow arrows
  right  45 s magnetogram (+-150 G) with the same annulus — MMFs are the
         small flux patches streaming outward through it
  bottom moat outflow curve with a time cursor

The spot is seeded where onset_analysis seeded it, and the bottom panel
is that run's curve (onset_series.npz) with its penumbra and moat onsets
and the literature formation interval -- run onset_analysis.py first.

Usage:
  python scripts/moat_movie.py AR11490 [--stride 8] [--fps 8] [--half 90]
  python scripts/moat_movie.py --all [--skip-existing]

stride 8 at 8 fps = 48 min of solar time per second of video.
"""

import argparse
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from moatflow.analysis.onsets import analyse
from moatflow.analysis.penumbra import segment_frame
from moatflow.analysis.spottrack import PX_MM, seed_epoch, track_spot
from moatflow.batch import CUBE_IC, FLOW_IC, add_event_args, \
    resolve_events, run_batch
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes, parse_times


MOVIE = "moat_tracking_explained.mp4"


def render(event_id, args, cfg):
    out = event_dir(cfg, event_id)
    npz_path = out / "onset_series.npz"
    if not npz_path.exists():
        raise SystemExit(f"{event_id}: no onset_series.npz -- run "
                         f"python scripts/onset_analysis.py {event_id} first")
    ic, t_iso, _ = load_cube(out / CUBE_IC)
    times_s = parse_times(t_iso)
    # magnetogram is optional; matched to the continuum by time, not index
    mg_file = out / "cube_hmi.M_45s_magnetogram.h5"
    mg = None
    if mg_file.exists():
        mg, t_mg_iso, _ = load_cube(mg_file)
        t0 = parse_datetimes(t_iso[:1])[0]
        t_mg = np.array([(t - t0).total_seconds()
                         for t in parse_datetimes(t_mg_iso)])
    else:
        print("  no magnetogram cube: right panel left empty")

    with h5py.File(out / FLOW_IC) as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]
    t_mid_h = t_mid / 3600

    d = np.load(npz_path)
    t_curve, v_curve = d["t_h"], d["v_gran"]
    on = analyse(d)

    # tracked spot geometry at the flow-window epochs, interpolated to
    # all frame times (curve holds through short tracking dropouts)
    epoch_idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    seed_h = seed_epoch(t_mid_h, on["lit_end_h"], override=args.seed_h,
                        npz_path=npz_path)
    tr = track_spot([np.asarray(ic[i]) for i in epoch_idx],
                    int(np.argmin(np.abs(t_mid_h - seed_h))))
    print(f"tracked {tr['valid'].sum()}/{len(epoch_idx)} epochs "
          f"(seed {t_mid_h[tr['seed_idx']]:.0f} h)")
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
    axC.plot(t_curve, v_curve, "-", color="tab:blue", lw=1.2,
             label="moat outflow")
    axC.axhline(0, color="k", lw=0.5)
    axA = axC.twinx()
    axA.plot(t_curve, d["a_pen"], "-", color="tab:orange", lw=1.0,
             label="penumbral area")
    axA.set_ylabel("penumbra [Mm$^2$]", color="tab:orange")
    lo, hi = t_curve[0], t_curve[-1]
    l0, l1 = on["lit_start_h"], on["lit_end_h"]
    if np.isfinite(l0) and np.isfinite(l1) and l1 > lo and l0 < hi:
        axC.axvspan(max(l0, lo), min(l1, hi), alpha=0.12, color="green",
                    label="literature formation")
    elif np.isfinite(l0) and not np.isfinite(l1) and lo <= l0 <= hi:
        axC.axvline(l0, color="green", lw=1.2, ls="-.",
                    label="literature formation start")
    for key, col, lab in (("t_pen_h", "tab:orange", "penumbra onset"),
                          ("t_moat_h", "tab:blue", "moat onset")):
        if np.isfinite(on.get(key, np.nan)):
            axC.axvline(on[key], color=col, ls="--", lw=1.1,
                        label=f"{lab} {on[key]:.1f} h")
    axC.set_xlim(lo, hi)
    axC.legend(loc="upper left", fontsize=7, ncol=2)
    axC.set(xlabel="hours since start", ylabel="moat outflow [km/s]")
    cursor = axC.axvline(0, color="crimson", lw=1.2)
    axC.grid(alpha=0.3)

    writer = FFMpegWriter(fps=args.fps, bitrate=4500)
    out_mp4 = out / "quicklook" / MOVIE
    out_mp4.parent.mkdir(exist_ok=True)
    print(f"rendering {len(range(0, len(ic), args.stride))} frames ...")

    with writer.saving(fig, str(out_mp4), dpi=110):
        for n in range(0, len(ic), args.stride):
            for ax in (axL, axR):
                ax.clear()
            frame_ic = np.asarray(ic[n])
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

            j = int(np.argmin(np.abs(t_mg - times_s[n]))) \
                if mg is not None else None
            if j is not None and abs(t_mg[j] - times_s[n]) <= 300:
                axR.imshow(np.asarray(mg[j]), **mg_kw)
                axR.set_title("magnetogram $\\pm$150 G — MMFs cross the "
                              "annulus")
            else:
                axR.imshow(np.zeros_like(frame_ic), cmap="gray", vmin=-1,
                           vmax=1, origin="lower")
                axR.set_title("no magnetogram cube" if mg is None
                              else "no magnetogram within 5 min")

            for ax in (axL, axR):
                for r, ls in ((r0, "--"), (r1, "--")):
                    ax.add_patch(Circle((c[1], c[0]), r, fill=False,
                                        color="crimson", ls=ls, lw=1.1))
                # same 10/20/30 Mm reference rings as the audit movie
                for r_mm in (10, 20, 30):
                    ax.add_patch(Circle((c[1], c[0]), r_mm / PX_MM,
                                        fill=False, color="white", ls=":",
                                        lw=1.4 if r_mm == 30 else 0.8,
                                        alpha=0.9 if r_mm == 30 else 0.55))
                ax.plot(c[1], c[0], "+", color="crimson", ms=10)
                ax.set_xlim(c[1] - args.half, c[1] + args.half)
                ax.set_ylim(c[0] - args.half, c[0] + args.half)
                ax.set_axis_off()

            t_h = times_s[n] / 3600
            cursor.set_xdata([t_h, t_h])
            fig.suptitle(f"{event_id} — moat tracking   "
                         f"{t_iso[n][:19]}  (t = {t_h:.1f} h)", y=0.98)
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out_mp4}")


def main():
    ap = argparse.ArgumentParser()
    add_event_args(ap)
    ap.add_argument("--stride", type=int, default=8,
                    help="use every Nth 45s frame (8 -> 6 min steps)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--half", type=int, default=90,
                    help="half-size of the zoom box [px]")
    ap.add_argument("--seed-h", type=float, default=None,
                    help="tracking seed [h]; default: the seed "
                         "onset_analysis used")
    args = ap.parse_args()

    cfg = load_config()
    ids = resolve_events(args, cfg, ap)
    sys.exit(run_batch(lambda e: render(e, args, cfg), ids, out_name=MOVIE,
                       cfg=cfg, skip_existing=args.skip_existing))


if __name__ == "__main__":
    main()
