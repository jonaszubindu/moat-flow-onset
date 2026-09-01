#!/usr/bin/env python
"""Minimal example: hmi.M_45s magnetograms cut to a SHARP's FOV.

Downloads the full-disk 45 s magnetograms as JSOC server-side `im_patch`
cutouts that track the active region, covering the SHARP's field of view
over its complete time series. Standalone — needs only `drms`, numpy,
pandas.

Gotchas (learned the hard way):
  * im_patch `t` is the NO-track flag: t=0 tracks the patch with solar
    rotation, t=1 leaves it fixed on the sky (AR drifts ~13 px/h).
  * `r=1` registers sub-pixel; without it the tracking advances in ~1 px
    window jumps (multi-km/s sawtooth artifacts in pairwise LCT).
  * SHARP keyword queries can return the string 'Invalid KeyLink'
    (CRSIZE1/2 does for entire HARPs) — coerce numerics and derive the
    FOV from LONDTMIN/MAX, LATDTMIN/MAX instead.
  * drms never overwrites: a re-download of an interrupted chunk lands
    as *.fits.1 next to the truncated original — hence the completeness
    check before a chunk is marked done.

Volume: a full HARP passage (~13 d for HARP 1701) at 45 s is ~23000
frames, ~10 GB rice-compressed at 622x338 px. Chunked + resumable:
rerun after any interruption and it continues where it stopped.
"""

from datetime import datetime, timedelta
from pathlib import Path

import drms
import numpy as np
import pandas as pd

EMAIL = "jonas.zbinden@unibe.ch"   # JSOC-registered
HARPNUM = 1701                     # e.g. AR 11490
OUT = Path(f"m45s_harp{HARPNUM}")
CHUNK_HOURS = 6
MARGIN = 1.1                       # FOV margin over the SHARP patch
PX_PER_DEG = 959.6 * np.pi / 180 / 0.504   # CCD px per heliogr. degree


def tai(t: datetime) -> str:
    return t.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def parse(t_rec: str) -> datetime:
    return datetime.strptime(str(t_rec)[:19], "%Y.%m.%d_%H:%M:%S")


client = drms.Client(email=EMAIL)

# --- SHARP metadata: full time series, FOV, patch anchor --------------
keys = client.query(f"hmi.sharp_cea_720s[{HARPNUM}][][? (QUALITY=0) ?]",
                    key="T_REC, LON_FWT, LAT_FWT, "
                        "LONDTMIN, LONDTMAX, LATDTMIN, LATDTMAX")
num = lambda col: pd.to_numeric(col, errors="coerce")
lon, lat = num(keys.LON_FWT), num(keys.LAT_FWT)
width = int((num(keys.LONDTMAX) - num(keys.LONDTMIN)).max()
            * PX_PER_DEG * MARGIN)
height = int((num(keys.LATDTMAX) - num(keys.LATDTMIN)).max()
             * PX_PER_DEG * MARGIN)
iref = int(np.nanargmin(np.abs(lon)))   # anchor: record nearest disk center

process = {"im_patch": {
    "t_ref": str(keys.T_REC[iref]),
    "t": 0,                       # 0 = track with solar rotation
    "r": 1,                       # register sub-pixel (no 1-px sawtooth)
    "c": 0,
    "locunits": "stony",          # x,y = Stonyhurst lon/lat in degrees
    "boxunits": "pixels",
    "x": float(lon[iref]), "y": float(lat[iref]),
    "width": width, "height": height,
}}
print(f"HARP {HARPNUM}: {keys.T_REC.iloc[0]} .. {keys.T_REC.iloc[-1]}, "
      f"patch {width}x{height} px anchored at "
      f"({lon[iref]:.2f}, {lat[iref]:.2f}) deg")

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
