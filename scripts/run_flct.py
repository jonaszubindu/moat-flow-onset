#!/usr/bin/env python
"""Build the HDF5 cube for one event/series and run windowed FLCT on it.

Usage:
  python scripts/run_flct.py AR11490 --series hmi.Ic_45s
  python scripts/run_flct.py AR11490 --series hmi.M_45s --window 7200
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.time import Time

from moatflow.config import event_dir, load_config
from moatflow.cubes import build_cube, load_cube
from moatflow.tracking.flct import flct_windowed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--series", default="hmi.Ic_45s")
    ap.add_argument("--window", type=float, default=3600.0,
                    help="averaging window [s]")
    ap.add_argument("--sigma", type=float, default=5.0,
                    help="FLCT Gaussian window [px] (~1.8 Mm at 5 px)")
    ap.add_argument("--stride", type=int, default=1,
                    help="frame separation of FLCT pairs")
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    cube_file = out / f"cube_{args.series}.h5"
    if not cube_file.exists():
        build_cube(out / args.series, cube_file)

    cube, t_iso, _ = load_cube(cube_file)
    t = Time([s.replace("_TAI", "").replace(".", "-").replace("_", "T")
              for s in t_iso], scale="tai")
    times_s = (t - t[0]).sec

    t_mid, vx, vy = flct_windowed(cube, np.asarray(times_s),
                                  window_s=args.window,
                                  pair_stride=args.stride,
                                  sigma_px=args.sigma)

    flow_file = out / f"flct_{args.series}_w{int(args.window)}s.h5"
    with h5py.File(flow_file, "w") as h5:
        h5["t_mid_s"] = t_mid
        h5["vx"] = vx
        h5["vy"] = vy
        h5.attrs.update({"window_s": args.window, "sigma_px": args.sigma,
                         "pair_stride": args.stride, "t0_isot": t[0].isot})
    print(f"Wrote {flow_file}: {len(t_mid)} flow maps")


if __name__ == "__main__":
    main()
