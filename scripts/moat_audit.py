#!/usr/bin/env python
"""Re-render the moat verification movie on its own.

`onset_analysis.py` already writes this movie for every event; use this
script to re-render with different options (zoom, pace, annulus
convention) without redoing the onset analysis. The spot is seeded where
onset_analysis seeded it, so the movie shows the track behind the numbers.

Usage:
  python scripts/moat_audit.py AR11210 [--annulus scaled] [--fps 6]
  python scripts/moat_audit.py --all [--skip-existing]
"""

import argparse
import sys
from datetime import datetime

import h5py
import numpy as np

from moatflow.analysis.audit import audit_series, render_audit_movie
from moatflow.analysis.spottrack import seed_epoch, track_spot
from moatflow.batch import CUBE_IC, FLOW_IC, add_event_args, \
    resolve_events, run_batch
from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes, parse_times


def render(event_id, args, cfg):
    out = event_dir(cfg, event_id)
    cube, t_iso, _ = load_cube(out / CUBE_IC)
    times_s = parse_times(t_iso)
    with h5py.File(out / FLOW_IC) as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]
    t_h = t_mid / 3600

    # literature penumbra-formation interval, in hours since window start
    ev = load_events()[event_id]
    t0 = parse_datetimes(t_iso[:1])[0]

    def lit(key):
        return ((datetime.fromisoformat(str(ev[key])) - t0).total_seconds()
                / 3600) if ev.get(key) else np.nan
    pf_lit = (lit("t_penumbra_start"), lit("t_penumbra_end"))

    idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    frames = [np.asarray(cube[i]) for i in idx]
    seed = seed_epoch(t_h, pf_lit[1], override=args.seed_h,
                      npz_path=out / "onset_series.npz")
    tr = track_spot(frames, int(np.argmin(np.abs(t_h - seed))),
                    prefer_west=ev.get("seed_prefer_west", True))
    print(f"tracked {tr['valid'].sum()}/{len(frames)} epochs "
          f"(seed {t_h[tr['seed_idx']]:.0f} h)")

    aud = audit_series(frames, tr, VX, VY, mode=args.annulus)
    mp4 = out / "quicklook" / movie_name(args)
    render_audit_movie(frames, tr, VX, VY, t_h, [t_iso[i] for i in idx],
                       aud, mp4, event_id=event_id, fps=args.fps,
                       stride=args.stride, half=args.half, pf_lit=pf_lit)
    print(f"wrote {mp4}")


def movie_name(args):
    suffix = "" if args.annulus == "fixed" else f"_{args.annulus}"
    return f"moat_audit{suffix}.mp4"


def main():
    ap = argparse.ArgumentParser()
    add_event_args(ap)
    ap.add_argument("--annulus", choices=["fixed", "scaled"], default="fixed")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--seed-h", type=float, default=None,
                    help="tracking seed [h]; default: the seed "
                         "onset_analysis used")
    args = ap.parse_args()

    cfg = load_config()
    ids = resolve_events(args, cfg, ap)
    sys.exit(run_batch(lambda e: render(e, args, cfg), ids,
                       out_name=movie_name(args), cfg=cfg,
                       skip_existing=args.skip_existing))


if __name__ == "__main__":
    main()
