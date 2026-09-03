"""Tracked 45 s cutouts of hmi.Ic_45s / hmi.M_45s via JSOC im_patch.

The im_patch processing runs server-side at JSOC: it extracts a patch
around a reference heliographic position and *tracks it with solar
rotation*, so the returned cutout series is co-registered in time —
SHARP-like geometry, but at the 45 s cadence needed for granulation LCT.

Long ranges are split into chunks (default 6 h) so individual export
requests stay small enough for the JSOC processing queue, and chunks whose
files already exist are skipped, making the download resumable.
"""

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import drms

from .sharps import jsoc_time

SERIES_SEGMENTS = {
    "hmi.Ic_45s": "continuum",
    "hmi.M_45s": "magnetogram",
    "hmi.V_45s": "Dopplergram",
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


def _clear_chunk(out_dir: Path, c0: datetime, c1: datetime) -> int:
    """Delete files (and .fits.N siblings) whose T_REC falls in [c0, c1).

    Only ever called for a chunk with no completion marker, i.e. one
    whose download was interrupted — completed chunks are skipped before
    this point, so finished data is never touched.
    """
    removed = 0
    for f in list(out_dir.glob("*.fits")) + list(out_dir.glob("*.fits.*")):
        m = re.search(r"(\d{8}_\d{6})_TAI", f.name)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        if c0 <= t < c1:
            f.unlink()
            removed += 1
    if removed:
        print(f"  cleared {removed} partial file(s) from the interrupted "
              f"chunk {c0:%Y-%m-%dT%H:%M}")
    return removed


def download_patches(event: dict, jsoc_email: str, out_dir: Path,
                     series: str = "hmi.Ic_45s",
                     width_px: int = 512, height_px: int = 512,
                     cadence: str = "45s", chunk_hours: float = 6.0,
                     retries: int = 3, retry_wait_s: float = 60.0) -> list[str]:
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
    failed: list[str] = []
    for c0, c1 in _chunks(t0, t1, chunk_hours):
        qstr = (f"{series}[{jsoc_time(c0.isoformat())}-"
                f"{jsoc_time(c1.isoformat())}@{cadence}]{{{segment}}}")
        marker = out_dir / f".done_{c0:%Y%m%dT%H%M}"
        if marker.exists():
            continue
        # A chunk without a marker was interrupted mid-download; clear its
        # leftovers first. drms NEVER overwrites — a redo would otherwise
        # land as *.fits.1 beside the truncated original, and the original
        # (bad) file is the one later globs pick up.
        _clear_chunk(out_dir, c0, c1)
        # Per-chunk retries so a transient network drop costs one attempt,
        # not the whole run.
        for attempt in range(1, retries + 1):
            try:
                print(f"Exporting {qstr}" +
                      (f" (attempt {attempt})" if attempt > 1 else ""))
                req = client.export(qstr, method="url", protocol="fits",
                                    process=process)
                req.wait()
                result = req.download(str(out_dir))
                got = [f for f in result["download"] if f is not None]
                # only mark complete if every exported record arrived —
                # a sleep/network cut can end download() with a partial
                # file list and no exception
                if len(got) < len(req.urls):
                    raise RuntimeError(
                        f"only {len(got)}/{len(req.urls)} files downloaded")
                downloaded += got
                marker.touch()
                break
            except Exception as e:
                print(f"  chunk {c0:%Y-%m-%dT%H:%M} attempt {attempt} "
                      f"failed: {e}")
                if attempt == retries:
                    failed.append(f"{c0:%Y-%m-%dT%H:%M}")
                else:
                    time.sleep(retry_wait_s)
    if failed:
        print(f"{len(failed)} chunk(s) failed after {retries} attempts: "
              f"{failed} — re-run the same command to retry them.")
    return downloaded
