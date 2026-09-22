#!/usr/bin/env python
"""Find (and optionally repair) truncated FITS in an event's downloads.

A transfer cut mid-file still counts as downloaded, so a damaged file can
sit on disk until the cube is built and fails there. This scans the raw
FITS of an event and, with --fix, deletes the damaged files together with
the completion marker of the chunk they belong to — so re-running the
download refetches exactly those chunks.

Usage:
  python scripts/verify_downloads.py AR11630                     # report
  python scripts/verify_downloads.py AR11630 --fix               # repair
  python scripts/verify_downloads.py AR11630 --series hmi.M_45s
"""

import argparse
import re
from datetime import datetime

from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.download.integrity import bad_fits
from moatflow.download.patches45s import SERIES_SEGMENTS, _chunks, _parse

T_REC = re.compile(r"(\d{8}_\d{6})_TAI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--series", nargs="*", default=list(SERIES_SEGMENTS))
    ap.add_argument("--fix", action="store_true",
                    help="delete damaged files and their chunk markers")
    args = ap.parse_args()

    cfg = load_config()
    out = event_dir(cfg, args.event_id)
    event = load_events()[args.event_id]
    chunk_hours = cfg.get("patch", {}).get("chunk_hours", 6)
    chunks = list(_chunks(_parse(event["t_start"]), _parse(event["t_end"]),
                          chunk_hours))
    total_bad = 0

    for series in args.series:
        d = out / series
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.fits"))
        if not files:
            continue
        print(f"{series}: checking {len(files)} files ...", flush=True)
        damaged = bad_fits(files, verbose=True)
        total_bad += len(damaged)
        if not damaged:
            print("  all readable")
            continue
        print(f"  {len(damaged)} damaged")
        if not args.fix:
            continue
        markers = set()
        for f in damaged:
            m = T_REC.search(f.name)
            if m:
                t = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
                for c0, c1 in chunks:
                    if c0 <= t < c1:
                        markers.add(d / f".done_{c0:%Y%m%dT%H%M}")
                        break
            f.unlink()
        for mk in markers:
            if mk.exists():
                mk.unlink()
        print(f"  removed {len(damaged)} file(s) and {len(markers)} chunk "
              f"marker(s) — re-run the download to refetch those chunks")

    if total_bad == 0:
        print("nothing to repair")
    elif not args.fix:
        print("re-run with --fix to remove them and their chunk markers")


if __name__ == "__main__":
    main()
