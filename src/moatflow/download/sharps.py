"""Download hmi.sharp_cea_720s segments for one event.

Uses drms export with protocol='fits' so the DB keywords (incl. WCS) are
merged into the FITS headers ('as-is' segment files carry almost no header)
— same approach as sst_hmi_align/hmi.py, applied to a full disk passage.
"""

import time
from pathlib import Path

import drms

DEFAULT_SEGMENTS = ("continuum", "magnetogram", "Dopplergram", "bitmap", "Br")


def jsoc_time(t: str) -> str:
    """ISO ('2012-05-28T00:00:00') or JSOC ('2012.05.28_00:00:00_TAI')
    input -> JSOC query time string. Already-JSOC strings pass through
    (naive .replace('T', '_') would mangle the 'TAI' suffix)."""
    if t.endswith("_TAI"):
        return t
    return t.replace("-", ".").replace("T", "_") + "_TAI"


def download_sharps(event: dict, jsoc_email: str, out_dir: Path,
                    segments: tuple = DEFAULT_SEGMENTS, retries: int = 3,
                    retry_wait_s: float = 60.0) -> list[str]:
    t0, t1 = jsoc_time(event["t_start"]), jsoc_time(event["t_end"])
    qstr = (f"hmi.sharp_cea_720s[{event['harpnum']}][{t0}-{t1}]"
            f"[? (QUALITY=0) ?]{{{','.join(segments)}}}")
    print(f"JSOC query: {qstr}")

    client = drms.Client(email=jsoc_email)
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSOC read timeouts are common on long transfers; retry the export
    # and fetch only what is still missing (the 45 s download does the
    # same per chunk)
    for attempt in range(1, retries + 1):
        try:
            req = client.export(qstr, method="url", protocol="fits")
            req.wait()
            expected = [out_dir / f for f in req.urls["filename"]]
            missing = [i for i, f in enumerate(expected) if not f.exists()]
            if not missing:
                print("All SHARP files already on disk, skipping download.")
                return [str(f) for f in expected]
            print(f"Downloading {len(missing)} of {len(expected)} files "
                  "(rest already on disk).")
            req.download(str(out_dir), index=missing)
            if all(f.exists() for f in expected):
                return [str(f) for f in expected]
            raise RuntimeError("download ended with files missing")
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  SHARP download attempt {attempt}/{retries} failed "
                  f"({type(e).__name__}: {e}); retrying in "
                  f"{retry_wait_s:.0f} s")
            time.sleep(retry_wait_s)
