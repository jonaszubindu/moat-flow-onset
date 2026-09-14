#!/usr/bin/env python
"""Re-run the FLCT verification checks for one event on their own.

onset_analysis.py already runs these for every event (divergence map,
Doppler cross-check, shrinking-Sun immunity — moatflow.analysis.verify).
Use this to re-run them without redoing the onset analysis. Position and
epochs come from the catalog and the event's own window; nothing needs
to be typed per event.

Usage: python scripts/verify_flct.py AR11490 [--seed-h 75]
"""

import argparse
import json
from datetime import datetime

import h5py
import numpy as np

from moatflow.analysis.audit import audit_series
from moatflow.analysis.spottrack import track_spot
from moatflow.analysis.verify import run_verification
from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes, parse_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--seed-h", type=float, default=None,
                    help="tracking seed / divergence epoch [h]; default as "
                         "onset_analysis: literature formation end + 8 h")
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    event = load_events()[args.event_id]
    cube, t_iso, _ = load_cube(out / "cube_hmi.Ic_45s_continuum.h5")
    times_s = parse_times(t_iso)
    t0 = parse_datetimes(t_iso[:1])[0]
    with h5py.File(out / "flct_hmi.Ic_45s_continuum_w3600s_s5px_k1.h5") as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]
    t_h = t_mid / 3600

    def lit_h(key):
        if not event.get(key):
            return np.nan
        return (datetime.fromisoformat(str(event[key])) - t0).total_seconds() / 3600
    pf = (lit_h("t_penumbra_start"), lit_h("t_penumbra_end"))
    seed_h = args.seed_h if args.seed_h is not None else \
        (pf[1] + 8 if np.isfinite(pf[1]) else 0.85 * t_h[-1])
    seed_h = min(seed_h, t_h[-1] - 2)

    idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    frames = [np.asarray(cube[i]) for i in idx]
    tr = track_spot(frames, int(np.argmin(np.abs(t_h - seed_h))))
    aud = audit_series(frames, tr, VX, VY)
    res = run_verification(out, args.event_id, event, frames, tr, VX, VY,
                           t_mid, t0, aud, seed_h, pf)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
