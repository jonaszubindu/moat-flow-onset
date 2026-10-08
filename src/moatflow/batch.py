"""Run a per-event script over several events.

Scripts take `event_ids` (one or more) or `--all`; `--all` means every
catalog event whose 45 s continuum cube and FLCT flows exist. One failing
event is reported and the rest still run.
"""

import time
import traceback

from .catalog import load_events
from .config import event_dir

CUBE_IC = "cube_hmi.Ic_45s_continuum.h5"
FLOW_IC = "flct_hmi.Ic_45s_continuum_w3600s_s5px_k1.h5"


def add_event_args(ap):
    ap.add_argument("event_ids", nargs="*", help="e.g. AR11490 AR11184")
    ap.add_argument("--all", action="store_true",
                    help="every event with a 45 s continuum cube and flows")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip events whose movie is already on disk")


def resolve_events(args, cfg, ap):
    if args.all:
        ids = [e for e in load_events()
               if (event_dir(cfg, e) / CUBE_IC).exists()
               and (event_dir(cfg, e) / FLOW_IC).exists()]
        skipped = sorted(set(load_events()) - set(ids))
        if skipped:
            print(f"--all: no cube/flows yet for {', '.join(skipped)}")
        return ids
    if not args.event_ids:
        ap.error("give one or more event ids, or --all")
    return args.event_ids


def run_batch(fn, ids, out_name=None, cfg=None, skip_existing=False):
    """fn(event_id) for each id; prints a summary and returns an exit code."""
    failed, skipped = {}, []
    for k, ev in enumerate(ids, 1):
        if skip_existing and out_name and \
                (event_dir(cfg, ev) / "quicklook" / out_name).exists():
            skipped.append(ev)
            continue
        print(f"\n=== {ev} ({k}/{len(ids)}) ===", flush=True)
        t = time.time()
        try:
            fn(ev)
            print(f"    {ev} done in {(time.time() - t) / 60:.1f} min")
        except (Exception, SystemExit) as e:
            if len(ids) == 1:
                raise
            traceback.print_exc()
            failed[ev] = f"{type(e).__name__}: {e}"
    if len(ids) > 1:
        done = len(ids) - len(failed) - len(skipped)
        print(f"\n{done}/{len(ids)} done"
              + (f", skipped (exists): {', '.join(skipped)}" if skipped else ""))
        for ev, msg in failed.items():
            print(f"  FAILED {ev}: {msg}")
    return 1 if failed else 0
