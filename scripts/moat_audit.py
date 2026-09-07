#!/usr/bin/env python
"""Re-render the moat verification movie on its own.

`onset_analysis.py` already writes this movie for every event; use this
script to re-render with different options (zoom, pace, annulus
convention) without redoing the onset analysis.

Usage: python scripts/moat_audit.py AR11210 [--annulus scaled] [--fps 6]
"""

import argparse

import h5py
import numpy as np

from datetime import datetime

from moatflow.analysis.audit import audit_series, render_audit_movie
from moatflow.catalog import load_events
from moatflow.analysis.spottrack import track_spot
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--annulus", choices=["fixed", "scaled"], default="fixed")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--seed-h", type=float, default=None)
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
    seed = args.seed_h if args.seed_h is not None else 0.85 * t_h[-1]
    tr = track_spot(frames, int(np.argmin(np.abs(t_h - seed))))

    # literature penumbra-formation interval, in hours since window start
    ev = load_events()[args.event_id]
    t0 = datetime.strptime(str(t_iso[0])[:19].replace("_", "T")
                           .replace(".", "-", 2), "%Y-%m-%dT%H:%M:%S")
    def lit(key):
        return ((datetime.fromisoformat(str(ev[key])) - t0).total_seconds()
                / 3600) if ev.get(key) else np.nan
    pf_lit = (lit("t_penumbra_start"), lit("t_penumbra_end"))

    aud = audit_series(frames, tr, VX, VY, mode=args.annulus)
    suffix = "" if args.annulus == "fixed" else f"_{args.annulus}"
    mp4 = out / "quicklook" / f"moat_audit{suffix}.mp4"
    render_audit_movie(frames, tr, VX, VY, t_h, [t_iso[i] for i in idx],
                       aud, mp4, event_id=args.event_id, fps=args.fps,
                       stride=args.stride, half=args.half, pf_lit=pf_lit)
    print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
