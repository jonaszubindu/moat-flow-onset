#!/usr/bin/env python
"""Prepare vetting material: SHARP continuum + quicklook for each event.

Downloads only the continuum segment (enough for the eyeball pass:
pore->spot transition, single trackable spot, baseline length) and
produces the quicklook movie / summary / drift products. Idempotent —
finished events are skipped, so re-running after an interruption is fine.

Usage: python scripts/vet_batch.py [--skip AR11490]
"""

import argparse
import subprocess
import sys

from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.download.sharps import download_sharps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=["AR11490"])
    args = ap.parse_args()

    cfg = load_config()
    events = load_events()
    failed = []
    for eid, event in events.items():
        if eid in args.skip:
            continue
        print(f"\n===== {eid} =====", flush=True)
        try:
            out = event_dir(cfg, eid)
            download_sharps(event, cfg["jsoc_email"], out / "sharp_cea_720s",
                            segments=("continuum",))
            subprocess.run(
                [sys.executable, "scripts/quicklook.py", eid,
                 "--series", "sharp_cea_720s", "--segment", "continuum"],
                check=True)
        except Exception as e:
            print(f"{eid} FAILED: {e}", flush=True)
            failed.append(eid)
    print(f"\ndone; {len(failed)} failed: {failed}" if failed
          else "\nall events prepared")


if __name__ == "__main__":
    main()
