"""Download hmi.sharp_cea_720s segments for one event.

Uses drms export with protocol='fits' so the DB keywords (incl. WCS) are
merged into the FITS headers ('as-is' segment files carry almost no header)
— same approach as sst_hmi_align/hmi.py, applied to a full disk passage.
"""

from pathlib import Path

import drms

DEFAULT_SEGMENTS = ("continuum", "magnetogram", "bitmap", "Br")


def jsoc_time(t: str) -> str:
    """ISO ('2012-05-28T00:00:00') or JSOC ('2012.05.28_00:00:00_TAI')
    input -> JSOC query time string. Already-JSOC strings pass through
    (naive .replace('T', '_') would mangle the 'TAI' suffix)."""
    if t.endswith("_TAI"):
        return t
    return t.replace("-", ".").replace("T", "_") + "_TAI"


def download_sharps(event: dict, jsoc_email: str, out_dir: Path,
                    segments: tuple = DEFAULT_SEGMENTS) -> list[str]:
    t0, t1 = jsoc_time(event["t_start"]), jsoc_time(event["t_end"])
    qstr = (f"hmi.sharp_cea_720s[{event['harpnum']}][{t0}-{t1}]"
            f"[? (QUALITY=0) ?]{{{','.join(segments)}}}")
    print(f"JSOC query: {qstr}")

    client = drms.Client(email=jsoc_email)
    req = client.export(qstr, method="url", protocol="fits")
    req.wait()

    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f for f in req.urls["filename"]]
    if all(f.exists() for f in expected):
        print("All SHARP files already on disk, skipping download.")
        return [str(f) for f in expected]

    result = req.download(str(out_dir))
    return list(result["download"])
