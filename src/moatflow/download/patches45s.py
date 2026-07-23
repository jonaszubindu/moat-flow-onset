"""Tracked 45 s cutouts of hmi.Ic_45s / hmi.M_45s via JSOC im_patch.

The im_patch processing runs server-side at JSOC: it extracts a patch
around a reference heliographic position and *tracks it with solar
rotation*, so the returned cutout series is co-registered in time —
SHARP-like geometry, but at the 45 s cadence needed for granulation LCT.

Long ranges are split into chunks (default 6 h) so individual export
requests stay small enough for the JSOC processing queue, and chunks whose
files already exist are skipped, making the download resumable.
"""

from datetime import datetime, timedelta
from pathlib import Path

import drms

from .sharps import jsoc_time

SERIES_SEGMENTS = {
    "hmi.Ic_45s": "continuum",
    "hmi.M_45s": "magnetogram",
}


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(
        t.replace(".", "-").replace("_TAI", "").replace("_", "T"))


def _chunks(t0: datetime, t1: datetime, hours: float):
    step = timedelta(hours=hours)
    t = t0
    while t < t1:
        yield t, min(t + step, t1)
        t += step


def download_patches(event: dict, jsoc_email: str, out_dir: Path,
                     series: str = "hmi.Ic_45s",
                     width_px: int = 512, height_px: int = 512,
                     cadence: str = "45s", chunk_hours: float = 6.0) -> list[str]:
    """Export im_patch tracked cutouts for one event and one 45 s series."""
    if series not in SERIES_SEGMENTS:
        raise ValueError(f"series must be one of {list(SERIES_SEGMENTS)}")
    segment = SERIES_SEGMENTS[series]

    process = {
        "im_patch": {
            # Reference time & Stonyhurst position from resolve_event().
            # 't' is JSOC's NoTrack flag: t=0 tracks the patch with solar
            # rotation (what we want); t=1 keeps it fixed on the sky and
            # the AR drifts through the box at ~13 px/h.
            # 'r' registers sub-pixel: without it the tracking advances in
            # discrete ~1 px window jumps — a sawtooth that puts multi-km/s
            # artifacts into pairwise LCT. Verified on AR11490: r=1 leaves
            # frame-to-frame steps of <0.03 px.
            "t_ref": _parse(event["t_ref"]).strftime("%Y.%m.%d_%H:%M:%S_TAI"),
            "t": 0,
            "r": 1,
            "c": 0,
            "locunits": "stony",
            "boxunits": "pixels",
            "x": event["lon_ref"],
            "y": event["lat_ref"],
            "width": width_px,
            "height": height_px,
        }
    }

    client = drms.Client(email=jsoc_email)
    out_dir = out_dir / series
    out_dir.mkdir(parents=True, exist_ok=True)

    t0, t1 = _parse(event["t_start"]), _parse(event["t_end"])
    downloaded: list[str] = []
    for c0, c1 in _chunks(t0, t1, chunk_hours):
        qstr = (f"{series}[{jsoc_time(c0.isoformat())}-"
                f"{jsoc_time(c1.isoformat())}@{cadence}]{{{segment}}}")
        marker = out_dir / f".done_{c0:%Y%m%dT%H%M}"
        if marker.exists():
            continue
        print(f"Exporting {qstr}")
        req = client.export(qstr, method="url", protocol="fits",
                            process=process)
        req.wait()
        result = req.download(str(out_dir))
        downloaded += list(result["download"])
        marker.touch()
    return downloaded
