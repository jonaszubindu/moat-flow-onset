#!/usr/bin/env python
"""Minimal example: hmi.M_45s magnetograms cut to a SHARP's FOV.

Downloads the full-disk 45 s magnetograms as JSOC server-side `im_patch`
cutouts that track the active region, covering the SHARP's field of view
over its complete time series. Standalone — needs only `drms` and numpy.

Gotchas (learned the hard way):
  * im_patch `t` is the NO-track flag: t=0 tracks the patch with solar
    rotation, t=1 leaves it fixed on the sky (AR drifts ~13 px/h).
  * `r=1` registers sub-pixel; without it the tracking advances in ~1 px
    window jumps (multi-km/s sawtooth artifacts in pairwise LCT).
  * drms never overwrites: a re-download of an interrupted chunk lands as
    *.fits.1 next to the truncated original — hence the completeness
    check before a chunk is marked done.

Volume: a full HARP passage (~10 d) at 45 s is ~19000 frames; at
~500x500 px that is roughly 20-30 GB. Chunked + resumable: rerun the
script after any interruption and it continues where it stopped.
"""

from datetime import datetime, timedelta
from pathlib import Path

import drms
import numpy as np

EMAIL = "jonas.zbinden@unibe.ch"   # JSOC-registered
HARPNUM = 1701                     # e.g. AR 11490
OUT = Path(f"m45s_harp{HARPNUM}")
CHUNK_HOURS = 6


def tai(t: datetime) -> str:
    return t.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def parse(t_rec: str) -> datetime:
    return datetime.strptime(str(t_rec)[:19], "%Y.%m.%d_%H:%M:%S")


client = drms.Client(email=EMAIL)

# --- SHARP metadata: full time series, FOV size, patch anchor ---------
keys = client.query(f"hmi.sharp_cea_720s[{HARPNUM}][][? (QUALITY=0) ?]",
                    key="T_REC, LON_FWT, LAT_FWT, CRSIZE1, CRSIZE2")
lon = np.asarray(keys.LON_FWT, float)
iref = int(np.nanargmin(np.abs(lon)))          # record nearest disk center
# CRSIZE is in CEA pixels (0.03 deg ~ 0.36"); used as CCD pixels (0.504")
# it covers the SHARP FOV with ~40% margin. Shrink if volume matters.
width = int(np.nanmax(keys.CRSIZE1))
height = int(np.nanmax(keys.CRSIZE2))

process = {"im_patch": {
    "t_ref": str(keys.T_REC[iref]),
    "t": 0,                       # 0 = track with solar rotation
    "r": 1,                       # register sub-pixel (no 1-px sawtooth)
    "c": 0,
    "locunits": "stony",          # x,y = Stonyhurst lon/lat in degrees
    "boxunits": "pixels",
    "x": float(keys.LON_FWT[iref]),
    "y": float(keys.LAT_FWT[iref]),
    "width": width, "height": height,
}}
print(f"HARP {HARPNUM}: {keys.T_REC.iloc[0]} .. {keys.T_REC.iloc[-1]}, "
      f"patch {width}x{height} px anchored at "
      f"({keys.LON_FWT[iref]:.1f}, {keys.LAT_FWT[iref]:.1f}) deg")

# --- chunked, resumable export ----------------------------------------
OUT.mkdir(exist_ok=True)
t, t_end = parse(keys.T_REC.iloc[0]), parse(keys.T_REC.iloc[-1])
while t < t_end:
    t_next = min(t + timedelta(hours=CHUNK_HOURS), t_end)
    marker = OUT / f".done_{t:%Y%m%dT%H%M}"
    if not marker.exists():
        query = f"hmi.M_45s[{tai(t)}-{tai(t_next)}@45s]{{magnetogram}}"
        print("Exporting", query, flush=True)
        req = client.export(query, method="url", protocol="fits",
                            process=process)
        req.wait()
        result = req.download(str(OUT))
        got = [f for f in result["download"] if f is not None]
        if len(got) < len(req.urls):
            raise RuntimeError(
                f"chunk incomplete ({len(got)}/{len(req.urls)}) — rerun")
        marker.touch()
    t = t_next
print("done:", len(list(OUT.glob("*.fits"))), "files in", OUT)
