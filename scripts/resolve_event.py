#!/usr/bin/env python
"""Fill harpnum / time window / im_patch reference of catalog events.

Usage: python scripts/resolve_event.py AR11490 [AR12345 ...]
       python scripts/resolve_event.py --all
"""

import argparse

from moatflow.catalog import load_events, resolve_event, save_events
from moatflow.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    events = load_events()
    ids = list(events) if args.all else args.event_ids
    for eid in ids:
        print(f"Resolving {eid} ...")
        events[eid] = resolve_event(events[eid], cfg["jsoc_email"],
                                    lon_max=cfg["max_abs_longitude_deg"])
        print(f"  HARP {events[eid]['harpnum']}: "
              f"{events[eid]['t_start']} - {events[eid]['t_end']}")
    save_events(events)


if __name__ == "__main__":
    main()
