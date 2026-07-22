"""Event catalog: load events.yaml, resolve NOAA -> HARP, fill time windows.

An event entry needs only `noaa_ar` (+ optionally the literature
penumbra-formation interval `t_penumbra_start`/`t_penumbra_end`).
`resolve_event` adds:
  harpnum      from the official JSOC NOAA<->HARP mapping
  t_start/t_end  download window: the literature interval padded by
                 (pad_before_h, pad_after_h) if given, otherwise the full
                 disk passage — in both cases clipped to
                 |Stonyhurst lon| <= lon_max
  t_ref, lon_ref, lat_ref  patch reference point for the im_patch export
It warns when the literature interval does not overlap the HARP's disk
passage (catches transcription errors in literature tables).
Resolved values are written back to events.yaml so JSOC is queried once.
"""

from datetime import datetime, timedelta
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _harp_noaa_table() -> pd.DataFrame:
    return pd.read_csv(HARP_NOAA_MAP_URL, sep=r"\s+")


def harp_for_noaa(noaa_ar: int) -> int:
    """HARPNUM containing a NOAA AR, from the official JSOC mapping table."""
    table = _harp_noaa_table()
    hits = [int(row.HARPNUM) for row in table.itertuples()
            if str(noaa_ar) in str(row.NOAA_ARS).split(",")]
    if not hits:
        raise ValueError(f"No HARP found for NOAA AR {noaa_ar}")
    if len(hits) > 1:
        print(f"NOAA {noaa_ar} maps to several HARPs {hits}; taking first.")
    return hits[0]


def _parse_time(t: str) -> datetime:
    """Parse ISO ('2012-05-28T14:00') or JSOC ('2012.05.28_14:00:00_TAI')."""
    t = str(t).replace("_TAI", "").replace(".", "-").replace("_", "T")
    return datetime.fromisoformat(t)


def _fmt_tai(t: datetime) -> str:
    return t.strftime("%Y.%m.%d_%H:%M:%S_TAI")


def resolve_event(event: dict, jsoc_email: str, lon_max: float = 40.0,
                  pad_before_h: float = 36.0,
                  pad_after_h: float = 24.0) -> dict:
    """Fill harpnum, time window and im_patch reference point of one event.

    Existing t_start/t_end in the entry are kept (manual override); only
    missing fields are computed. If the entry carries a literature
    penumbra-formation interval (t_penumbra_start / t_penumbra_end, UT),
    the window is that interval padded by pad_before_h / pad_after_h;
    otherwise it is the full |lon| <= lon_max disk passage.
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

    passage0 = _parse_time(str(keys.T_REC[first]))
    passage1 = _parse_time(str(keys.T_REC[last]))

    w0, w1 = passage0, passage1
    if event.get("t_penumbra_start"):
        pf0 = _parse_time(event["t_penumbra_start"])
        pf1 = _parse_time(event.get("t_penumbra_end")
                          or event["t_penumbra_start"])
        if pf1 < passage0 or pf0 > passage1:
            print(f"  WARNING: literature interval {pf0} - {pf1} does not "
                  f"overlap the disk passage {passage0} - {passage1} of "
                  f"HARP {event['harpnum']} (|lon| <= {lon_max} deg) — "
                  "check the transcribed table dates; using full passage.")
        else:
            w0 = max(pf0 - timedelta(hours=pad_before_h), passage0)
            w1 = min(pf1 + timedelta(hours=pad_after_h), passage1)

    if event.get("t_start") is None:
        event["t_start"] = _fmt_tai(w0)
    if event.get("t_end") is None:
        event["t_end"] = _fmt_tai(w1)

    # Patch reference: flux-weighted AR position at the record closest to
    # central meridian, so the im_patch stays centered on the spot group.
    # Restrict to records inside the longitude cut (LON_FWT can be NaN,
    # and np.argmin would return the first NaN index).
    idx_ok = np.flatnonzero(ok)
    iref = int(idx_ok[np.nanargmin(np.abs(lon[idx_ok]))])
    event["t_ref"] = str(keys.T_REC[iref])
    event["lon_ref"] = float(keys.LON_FWT[iref])
    event["lat_ref"] = float(keys.LAT_FWT[iref])
    return event
