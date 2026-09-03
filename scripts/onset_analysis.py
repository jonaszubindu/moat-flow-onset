#!/usr/bin/env python
"""Per-event onset comparison: penumbral area vs moat outflow (both tracers).

Produces  data/<event>/onset_series.npz  and
          data/<event>/quicklook/onset_comparison.png
with the tracked leading spot's penumbral/umbral areas, the granulation
moat-outflow curve, the MMF curve (if magnetogram flows exist), the
literature formation interval band, and provisional onset readouts.

Usage: python scripts/onset_analysis.py AR11184 [--seed-h 116] [--vthresh 0.15]
"""

import argparse
from datetime import datetime

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moatflow.analysis.spottrack import PX_MM, track_spot
from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes, parse_times

GAP_MM, WIDTH_MM = 1.0, 6.0


def moat_curve(vx_all, vy_all, tr, shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    v = np.full(len(vx_all), np.nan)
    for k in range(len(vx_all)):
        if not tr["valid"][k]:
            continue
        cy, cx = tr["center"][k]
        rr = np.hypot(xx - cx, yy - cy)
        r0 = tr["r_spot_px"][k] + GAP_MM / PX_MM
        sel = (rr >= r0) & (rr < r0 + WIDTH_MM / PX_MM)
        with np.errstate(invalid="ignore"):
            vr = ((vx_all[k] * (xx - cx) + vy_all[k] * (yy - cy))
                  / np.where(rr > 0, rr, np.nan))
        good = sel & np.isfinite(vr)
        if good.sum() > 30:
            v[k] = np.nanmean(vr[good])
    return v


def onset(t_h, v, thresh, persist=3):
    ok = np.nan_to_num(v) > thresh
    for i in range(len(ok) - persist + 1):
        if ok[i:i + persist].all():
            return float(t_h[i])
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--seed-h", type=float, default=None,
                    help="tracking seed epoch [h]; default: literature "
                         "formation end + 8 h")
    ap.add_argument("--vthresh", type=float, default=0.15)
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    event = load_events()[args.event_id]

    cube_file = out / "cube_hmi.Ic_45s_continuum.h5"
    flow_file = out / "flct_hmi.Ic_45s_continuum_w3600s_s5px_k1.h5"
    if not cube_file.exists() or not flow_file.exists():
        raw = out / "hmi.Ic_45s"
        n_fits = len(list(raw.glob("*.fits"))) if raw.is_dir() else 0
        msg = [f"{args.event_id}: cannot run the onset analysis yet."]
        if n_fits == 0 and not cube_file.exists():
            msg += [
                "  No 45 s continuum data (only the SHARP vetting material?).",
                f"  1) python scripts/download_event.py {args.event_id} "
                "--patches --series hmi.Ic_45s",
                f"  2) python scripts/quicklook.py {args.event_id} "
                "--series hmi.Ic_45s        # builds the cube + vets drift",
                f"  3) python scripts/run_flct.py {args.event_id} "
                "--series hmi.Ic_45s --window 3600 --sigma 5",
            ]
        elif not cube_file.exists():
            msg += [f"  {n_fits} FITS on disk but no cube yet:",
                    f"  python scripts/quicklook.py {args.event_id} "
                    "--series hmi.Ic_45s"]
        else:
            msg += ["  Cube exists but no flow maps yet:",
                    f"  python scripts/run_flct.py {args.event_id} "
                    "--series hmi.Ic_45s --window 3600 --sigma 5"]
        msg += ["  (optional, for the MMF tracer: same steps with "
                "--series hmi.M_45s, FLCT with --thresh 30)"]
        raise SystemExit("\n".join(msg))

    cube, t_iso, _ = load_cube(cube_file)
    times_s = parse_times(t_iso)
    t0 = parse_datetimes(t_iso[:1])[0]

    def lit_h(key):
        if not event.get(key):
            return np.nan
        return (datetime.fromisoformat(str(event[key]))
                - t0).total_seconds() / 3600

    pf0, pf1 = lit_h("t_penumbra_start"), lit_h("t_penumbra_end")

    with h5py.File(flow_file) as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]
    t_h = t_mid / 3600

    seed_h = args.seed_h if args.seed_h is not None else \
        (pf1 + 8 if np.isfinite(pf1) else 0.85 * t_h[-1])
    seed_h = min(seed_h, t_h[-1] - 2)

    epoch_idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    frames = [np.asarray(cube[i]) for i in epoch_idx]
    tr = track_spot(frames, int(np.argmin(np.abs(t_h - seed_h))))
    print(f"tracked {tr['valid'].sum()}/{len(frames)} epochs "
          f"(seed {seed_h:.0f} h)")

    v_gran = moat_curve(VX, VY, tr, frames[0].shape)
    mfile = out / "flct_hmi.M_45s_magnetogram_w3600s_s5px_k1.h5"
    v_mmf = None
    if mfile.exists():
        with h5py.File(mfile) as f:
            v_mmf = moat_curve(f["vx"][:], f["vy"][:], tr, frames[0].shape)

    np.savez(out / "onset_series.npz", t_h=t_h,
             a_pen=tr["area_penumbra"], a_umb=tr["area_umbra"],
             v_gran=v_gran,
             v_mmf=v_mmf if v_mmf is not None else np.full_like(v_gran, np.nan),
             valid=tr["valid"], pf_lit=(pf0, pf1))

    kern = np.ones(3) / 3
    sm = lambda v: np.convolve(np.nan_to_num(v), kern, "same")
    fig, ax1 = plt.subplots(figsize=(11, 5.4))
    ax1.plot(t_h, tr["area_penumbra"], "o-", ms=3, color="tab:orange",
             label="penumbral area (tracked spot)")
    ax1.plot(t_h, tr["area_umbra"], "s-", ms=2, alpha=0.5, color="tab:red",
             label="umbral area")
    ax1.set_xlabel(f"hours since {t_iso[0][:16]}")
    ax1.set_ylabel("area [Mm$^2$]", color="tab:orange")
    ax2 = ax1.twinx()
    ax2.plot(t_h, sm(v_gran), "o-", ms=3, color="tab:blue",
             label="moat outflow, granulation")
    if v_mmf is not None:
        ax2.plot(t_h, sm(v_mmf), "s-", ms=3, color="tab:purple",
                 label="moat outflow, MMF (valid post-penumbra)")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axhline(args.vthresh, color="gray", lw=0.5, ls="--")
    ax2.set_ylabel("moat outflow [km/s] (3h smooth)", color="tab:blue")
    if np.isfinite(pf0):
        ax1.axvspan(pf0, min(pf1, t_h[-1]), alpha=0.12, color="green",
                    label="literature penumbra formation")
    bad = ~tr["valid"]
    if bad.any():
        ax1.plot(t_h[bad], np.zeros(bad.sum()), "kx", ms=4,
                 label="tracking lost")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax2.legend(loc="lower right", fontsize=8.5)
    ax1.set_title(f"{args.event_id}: penumbra formation vs moat-flow onset")
    fig.tight_layout()
    fig.savefig(out / "quicklook" / "onset_comparison.png", dpi=140)

    on_g = onset(t_h, sm(v_gran), args.vthresh)
    on_m = onset(t_h, sm(v_mmf), args.vthresh) if v_mmf is not None else np.nan
    print(f"literature formation interval: {pf0:.1f} - {pf1:.1f} h")
    print(f"provisional moat onsets: granulation {on_g:.1f} h, MMF {on_m:.1f} h")
    print(f"plateau (last 20 h): gran {np.nanmean(sm(v_gran)[t_h > t_h[-1]-20]):.2f} km/s")


if __name__ == "__main__":
    main()
