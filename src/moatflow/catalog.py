"""Event catalog: load events.yaml, resolve NOAA -> HARP, fill time windows.

An event entry needs only `noaa_ar`. `resolve_event` adds:
  harpnum      from the official JSOC NOAA<->HARP mapping
  t_start/t_end  trimmed disk passage where |Stonyhurst lon| <= lon_max
  t_ref, lon_ref, lat_ref  patch reference point for the im_patch export
Resolved values are written back to events.yaml so JSOC is queried once.
"""

from pathlib import Path

import drms
import numpy as np
import pandas as pd
import yaml

from .config import REPO_ROOT

EVENTS_FILE = REPO_ROOT / "catalog" / "events.yaml"
HARP_NOAA_MAP_URL = ("http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/"
                     "all_harps_with_noaa_ars.txt")


def load_events(path: Path = EVENTS_FILE) -> dict:
    with open(path) as f:
        events = yaml.safe_load(f)["events"]
    return {e["id"]: e for e in events}


def save_events(events: dict, path: Path = EVENTS_FILE) -> None:
    with open(path, "w") as f:
        yaml.safe_dump({"events": list(events.values())}, f,
                       sort_keys=False, default_flow_style=False)


def harp_for_noaa(noaa_ar: int) -> int:
    """HARPNUM containing a NOAA AR, from the official JSOC mapping table."""
    table = pd.read_csv(HARP_NOAA_MAP_URL, sep=r"\s+")
    hits = [int(row.HARPNUM) for row in table.itertuples()
            if str(noaa_ar) in str(row.NOAA_ARS).split(",")]
    if not hits:
        raise ValueError(f"No HARP found for NOAA AR {noaa_ar}")
    if len(hits) > 1:
        print(f"NOAA {noaa_ar} maps to several HARPs {hits}; taking first.")
    return hits[0]


def resolve_event(event: dict, jsoc_email: str,
                  lon_max: float = 40.0) -> dict:
    """Fill harpnum, time window and im_patch reference point of one event.

    Existing t_start/t_end in the entry are kept (manual override); only
    missing fields are computed.
    """
    if event.get("harpnum") is None:
        event["harpnum"] = harp_for_noaa(event["noaa_ar"])

    client = drms.Client(email=jsoc_email)
    keys = client.query(
        f"hmi.sharp_cea_720s[{event['harpnum']}][][? (QUALITY=0) ?]",
        key="T_REC, LON_FWT, LAT_FWT, CRSIZE1, CRSIZE2, USFLUX, NOAA_AR")
    if len(keys) == 0:
        raise ValueError(f"No SHARP records for HARP {event['harpnum']}")

    lon = np.asarray(keys.LON_FWT, dtype=float)
    ok = np.abs(lon) <= lon_max
    if not ok.any():
        raise ValueError(
            f"HARP {event['harpnum']} never within |lon| <= {lon_max} deg")
    first, last = np.flatnonzero(ok)[[0, -1]]

    event.setdefault("t_start", str(keys.T_REC[first]))
    event.setdefault("t_end", str(keys.T_REC[last]))

    # Patch reference: flux-weighted AR position at the record closest to
    # central meridian, so the im_patch stays centered on the spot group.
    iref = int(np.argmin(np.abs(lon)))
    event["t_ref"] = str(keys.T_REC[iref])
    event["lon_ref"] = float(keys.LON_FWT[iref])
    event["lat_ref"] = float(keys.LAT_FWT[iref])
    return event
