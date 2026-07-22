#!/usr/bin/env python
"""Download SHARPs and/or 45 s tracked cutouts for one event.

Usage:
  python scripts/download_event.py AR11490 --sharps
  python scripts/download_event.py AR11490 --patches            # both 45s series
  python scripts/download_event.py AR11490 --patches --series hmi.Ic_45s
"""

import argparse

from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.download.patches45s import SERIES_SEGMENTS, download_patches
from moatflow.download.sharps import download_sharps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--sharps", action="store_true")
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--series", choices=list(SERIES_SEGMENTS),
                    help="restrict --patches to one 45s series")
    args = ap.parse_args()

    cfg = load_config()
    event = load_events()[args.event_id]
    if event.get("harpnum") is None or event.get("t_start") is None:
        raise SystemExit(f"{args.event_id} not resolved yet — "
                         "run scripts/resolve_event.py first.")
    out = event_dir(cfg, args.event_id)

    if args.sharps:
        download_sharps(event, cfg["jsoc_email"], out / "sharp_cea_720s")

    if args.patches:
        series_list = [args.series] if args.series else list(SERIES_SEGMENTS)
        p = cfg["patch"]
        for series in series_list:
            download_patches(event, cfg["jsoc_email"], out,
                             series=series,
                             width_px=p["width_px"], height_px=p["height_px"],
                             cadence=p["cadence"],
                             chunk_hours=p["chunk_hours"])


if __name__ == "__main__":
    main()
