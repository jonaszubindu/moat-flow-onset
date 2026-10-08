#!/usr/bin/env python
"""Compact inventory of everything the pipeline has produced, per event.

For every event directory under data_root it prints:
  raw    FITS per series: count, completed chunks vs the number the
         catalog window implies, duplicate (.fits.N) files, total size
  cube   frames, frame shape, and whether every frame has a timestamp
  flow   FLCT flow files and their number of maps
  files  onset / audit / verification products in the event root
  quick  the quicklook figures and movies
  check  a one-line summary of onset_series.npz and verify_flct.json
plus any interrupted builds (*.partial) it finds.

Raw FITS are counted, not listed, so the output stays short enough to
paste: about 15 lines per event.

Usage:
  python scripts/inventory.py                  # all events
  python scripts/inventory.py AR11630 AR11640  # selected events
  python scripts/inventory.py > inventory.txt
"""

import argparse
import json
import os
import socket
import subprocess
import textwrap
from datetime import datetime

import h5py
import numpy as np

from moatflow.catalog import load_events
from moatflow.config import load_config
from moatflow.cubes import cube_ok
from moatflow.download.patches45s import SERIES_SEGMENTS, _chunks, _parse


def human(n):
    for unit in "BKMGT":
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


def scan_raw(d):
    """(n_fits, n_duplicate, n_markers, bytes) for one raw-series directory."""
    n_fits = n_dup = n_markers = size = 0
    with os.scandir(d) as it:
        for e in it:
            name = e.name
            if name.startswith(".done_"):
                n_markers += 1
                continue
            if name.endswith(".fits"):
                n_fits += 1
            elif ".fits." in name:
                n_dup += 1
            else:
                continue
            try:
                size += e.stat().st_size
            except OSError:
                pass
    return n_fits, n_dup, n_markers, size


def cube_line(p):
    try:
        with h5py.File(p, "r") as h5:
            shape = h5["data"].shape
            dp = "  deprojected" if h5.attrs.get("DEPROJECTED", 0) else ""
        state = "complete" if cube_ok(p) else "INCOMPLETE"
        return f"{shape[0]} frames {shape[1]}x{shape[2]}  {state}{dp}"
    except Exception as e:
        return f"UNREADABLE ({type(e).__name__})"


def flow_line(p):
    try:
        with h5py.File(p, "r") as h5:
            return f"{h5['vx'].shape[0]} maps"
    except Exception as e:
        return f"UNREADABLE ({type(e).__name__})"


def onset_summary(p):
    try:
        d = np.load(p, allow_pickle=False)
        valid = d["valid"] if "valid" in d else None
        parts = [f"{int(valid.sum())}/{len(valid)} epochs tracked"
                 if valid is not None else "no 'valid' array"]
        if "v_mmf" in d:
            parts.append("MMF curve " +
                         ("yes" if np.isfinite(d["v_mmf"]).any() else "no"))
        if "annulus_mode" in d:
            parts.append(f"annulus {d['annulus_mode']}")
        return ", ".join(parts)
    except Exception as e:
        return f"UNREADABLE ({type(e).__name__})"


def verify_summary(p):
    try:
        r = json.loads(p.read_text())
        parts = []
        dv, dp, ss = r.get("divergence"), r.get("doppler"), r.get("shrinking_sun")
        if dv:
            parts.append(f"div {dv['annulus_mean_div_1e-5_per_s']:+.2f}e-5/s")
        parts.append(f"Doppler r={dp['r']:.2f} amp={dp['flct_over_doppler']:.2f}"
                     if dp else "Doppler skipped")
        if ss:
            parts.append(f"shrink-Sun shift {ss['max_curve_shift_m_s']:.1f} m/s")
        return ", ".join(parts)
    except Exception as e:
        return f"UNREADABLE ({type(e).__name__})"


def expected_chunks(event, chunk_hours):
    try:
        return len(list(_chunks(_parse(event["t_start"]),
                                _parse(event["t_end"]), chunk_hours)))
    except Exception:
        return None


def report_event(eid, d, event, chunk_hours):
    n_exp = expected_chunks(event, chunk_hours) if event else None
    head = f"=== {eid}"
    if event:
        head += f"  (catalog window {event.get('t_start')} .. {event.get('t_end')}"
        head += f", {n_exp} chunks)" if n_exp else ")"
    else:
        head += "  (not in catalog)"
    print(head)

    for series in list(SERIES_SEGMENTS) + ["sharp_cea_720s"]:
        sd = d / series
        if not sd.is_dir():
            print(f"  raw   {series:16s} -")
            continue
        n, dup, mk, size = scan_raw(sd)
        line = f"  raw   {series:16s} {n:6d} fits  {human(size):>7s}"
        if series in SERIES_SEGMENTS:
            line += f"  chunks {mk}/{n_exp if n_exp else '?'}"
        if dup:
            line += f"  {dup} DUPLICATE .fits.N"
        print(line)

    for p in sorted(d.glob("cube_*.h5")):
        print(f"  cube  {p.name[5:-3]:34s} {cube_line(p)}  {human(p.stat().st_size)}")
    for p in sorted(d.glob("flct_*.h5")):
        print(f"  flow  {p.name[5:-3]:34s} {flow_line(p)}  {human(p.stat().st_size)}")

    others = [p for p in sorted(d.iterdir())
              if p.is_file() and not p.name.startswith(".")
              and not p.name.startswith(("cube_", "flct_"))]
    if others:
        print("  files " + ", ".join(f"{p.name} ({human(p.stat().st_size)})"
                                     for p in others))

    ql = d / "quicklook"
    if ql.is_dir():
        names = sorted(p.name for p in ql.iterdir() if p.is_file())
        text = ", ".join(names) if names else "(empty)"
        print(textwrap.fill(text, width=100, initial_indent="  quick ",
                            subsequent_indent="        "))
    else:
        print("  quick -")

    if (d / "onset_series.npz").exists():
        print(f"  check onset_series : {onset_summary(d / 'onset_series.npz')}")
    if (ql / "verify_flct.json").exists():
        print(f"  check verify_flct  : {verify_summary(ql / 'verify_flct.json')}")

    partial = sorted(d.rglob("*.partial"))
    if partial:
        print("  WARN  interrupted builds: " + ", ".join(p.name for p in partial))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="*", help="event ids (default: all)")
    args = ap.parse_args()

    cfg = load_config()
    root = cfg["data_root"]
    catalog = load_events()
    chunk_hours = cfg.get("patch", {}).get("chunk_hours", 6)

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=cfg["repo_root"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or "?"
    except Exception:
        commit = "?"
    print(f"inventory  {datetime.now():%Y-%m-%d %H:%M}  host {socket.gethostname()}"
          f"  code {commit}")
    print(f"data_root  {root}")

    on_disk = {p.name for p in root.iterdir() if p.is_dir()} if root.is_dir() else set()
    ids = args.events or (list(catalog) + sorted(on_disk - set(catalog)))
    for eid in ids:
        d = root / eid
        if not d.is_dir():
            print(f"=== {eid}  (no data directory)")
            continue
        report_event(eid, d, catalog.get(eid), chunk_hours)


if __name__ == "__main__":
    main()
