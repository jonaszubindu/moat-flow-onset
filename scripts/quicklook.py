#!/usr/bin/env python
"""Vet downloaded data for one event: cube, movie, drift check, cadence.

Usage:
  python scripts/quicklook.py AR11490 --series sharp_cea_720s --segment continuum
  python scripts/quicklook.py AR11490 --series hmi.Ic_45s
  python scripts/quicklook.py AR11490 --series hmi.M_45s

Outputs into data/<event>/quicklook/:
  <tag>.mp4          movie (subsampled)
  <tag>_summary.png  first/mid/last frames + frame-mean series
  <tag>_drift.png    cumulative tracking drift [px] — should stay well
                     below ~1 px over the series; a linear trend means
                     the patch tracking leaks a bias flow into the LCT.
"""

import argparse

import numpy as np

from moatflow.config import event_dir, load_config
from moatflow.cubes import build_cube, load_cube
from moatflow.download.patches45s import SERIES_SEGMENTS
from moatflow.viz import (cadence_report, drift_series, parse_times,
                          plot_drift, save_movie, summary_figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--series", default="hmi.Ic_45s",
                    help="hmi.Ic_45s | hmi.M_45s | sharp_cea_720s")
    ap.add_argument("--segment", default=None,
                    help="segment (needed for sharp_cea_720s, e.g. continuum)")
    ap.add_argument("--drift-stride", type=int, default=None,
                    help="register every Nth frame (default: ~200 steps)")
    args = ap.parse_args()

    segment = args.segment or SERIES_SEGMENTS.get(args.series)
    if segment is None:
        raise SystemExit("--segment is required for this series")

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    ql = out / "quicklook"
    ql.mkdir(exist_ok=True)
    tag = f"{args.series}_{segment}"

    cube_file = out / f"cube_{tag}.h5"
    if not cube_file.exists():
        build_cube(out / args.series, cube_file, pattern=f"*.{segment}.fits")
    cube, t_iso, _ = load_cube(cube_file)

    times_s = parse_times(t_iso)
    print(f"{args.event_id} / {tag}:")
    cadence_report(times_s)

    stride = args.drift_stride or max(1, len(cube) // 200)
    drift = drift_series(cube, stride=stride)
    t_drift = times_s[::stride][:len(drift)]
    plot_drift(drift, t_drift / 3600,
               ql / f"{tag}_drift.png", title=f"{args.event_id} {tag}")
    # Bias velocity, not cumulative px: whole-scene registration includes
    # the AR's real proper motion, so this is an upper bound on tracking
    # error. Compare against the ~0.5-1 km/s moat signal.
    HMI_PIXEL_KM = 362.0
    span = max(float(t_drift[-1] - t_drift[0]), 1.0)
    v_bias = np.hypot(*(drift[-1] - drift[0])) * HMI_PIXEL_KM / span
    # worst 6-h window catches transient tracking glitches a net rate hides
    n6 = max(2, int(round(6 * 3600 / (span / max(len(drift) - 1, 1)))))
    v6 = max((np.hypot(*(drift[i + n6] - drift[i])) * HMI_PIXEL_KM
              / (t_drift[i + n6] - t_drift[i])
              for i in range(len(drift) - n6)), default=v_bias)
    print(f"  scene drift: net {v_bias:.3f} km/s, worst 6 h {v6:.3f} km/s "
          f"({'OK, well below moat ~0.5 km/s' if v6 < 0.1 else 'CHECK TRACKING/DRIFT PLOT'})")

    summary_figure(cube, times_s / 3600, ql / f"{tag}_summary.png",
                   segment=segment, title=f"{args.event_id} {tag}")
    save_movie(cube, t_iso, ql / f"{tag}.mp4", segment=segment,
               title=args.event_id)
    print(f"  quicklook products in {ql}")


if __name__ == "__main__":
    main()
