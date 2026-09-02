"""Assemble downloaded FITS series into one HDF5 cube per event/series.

Layout (one file, e.g. AR11490/cube_hmi.Ic_45s_continuum.h5):
    data      (nt, ny, nx)  float32
    t_obs     (nt,)         ISO-8601 strings (TAI)
    attrs     reference FITS header of the first frame (patch WCS)

Half a million individual FITS files do not survive contact with a
cluster filesystem; the cubes are what tracking and analysis read.
"""

import glob
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def _first_image_hdu(hdul):
    return next(h for h in hdul if h.data is not None)


def _trec_digits(s: str) -> str:
    """Normalize any T_REC representation to its digit string."""
    return "".join(ch for ch in str(s) if ch.isdigit())


def jsoc_bad_trecs(series: str, files: list, email: str) -> set[str]:
    """T_RECs (digit-normalized) with QUALITY != 0 over the files' span.

    The 45 s im_patch exports select records by TIME only — unlike our
    SHARP queries there is no QUALITY filter, so eclipse-season or
    calibration frames arrive as normal-looking FITS. Query the series
    keywords over the downloaded span and return the bad set so cube
    building can exclude them.
    """
    import re

    import drms

    trecs = sorted(_trec_digits(re.search(r"(\d{8}_\d{6})_TAI",
                                          os.path.basename(f)).group(1))
                   for f in files)
    t0, t1 = trecs[0], trecs[-1]
    fmt = lambda d: (f"{d[0:4]}.{d[4:6]}.{d[6:8]}_"
                     f"{d[8:10]}:{d[10:12]}:{d[12:14]}_TAI")
    k = drms.Client(email=email).query(
        f"{series}[{fmt(t0)}-{fmt(t1)}@45s]", key="T_REC, QUALITY")
    bad = {_trec_digits(t) for t, q in zip(k.T_REC, k.QUALITY.astype(str))
           if q not in ("0x00000000", "0")}
    return bad


def build_cube(fits_dir: Path, out_file: Path,
               pattern: str = "*.fits",
               exclude_trecs: set[str] | None = None) -> Path:
    import re

    files = sorted(glob.glob(str(fits_dir / pattern)))
    if exclude_trecs:
        n0 = len(files)
        files = [f for f in files
                 if _trec_digits(re.search(r"(\d{8}_\d{6})_TAI",
                                           os.path.basename(f)).group(1))
                 not in exclude_trecs]
        print(f"QC: excluded {n0 - len(files)} QUALITY!=0 frames")
    if not files:
        raise FileNotFoundError(f"No FITS files in {fits_dir}")

    with fits.open(files[0]) as hdul:
        hdu = _first_image_hdu(hdul)
        shape, header = hdu.data.shape, hdu.header

    with h5py.File(out_file, "w") as h5:
        data = h5.create_dataset("data", shape=(len(files), *shape),
                                 dtype="f4", chunks=(1, *shape))
        t_obs = h5.create_dataset("t_obs", shape=(len(files),),
                                  dtype=h5py.string_dtype())
        for i, f in enumerate(files):
            # surface the offending file — truncated FITS from interrupted
            # downloads otherwise fail deep inside astropy
            try:
                with fits.open(f) as hdul:
                    hdu = _first_image_hdu(hdul)
                    if hdu.data.shape != shape:
                        raise ValueError(
                            f"shape {hdu.data.shape} != {shape} — "
                            "frame not tracked/cropped consistently?")
                    data[i] = hdu.data.astype("f4")
                    t = hdu.header.get("T_OBS") or hdu.header.get("DATE-OBS")
                    t_obs[i] = t
            except Exception as e:
                raise RuntimeError(f"unreadable FITS {f}: {e}") from e
        for k, v in header.items():
            if k and not isinstance(v, fits.header._HeaderCommentaryCards):
                try:
                    h5.attrs[k] = v
                except TypeError:
                    pass
    print(f"Wrote {out_file}: {len(files)} frames of {shape}")
    return out_file


def build_event_cube(event_dir: Path, series: str, segment: str,
                     email: str | None = None) -> Path:
    """Build (or reuse) the cube for one event/series, QUALITY-filtered.

    For the 45 s series the im_patch export has no QUALITY filter, so
    bad records are fetched from JSOC and excluded here; SHARP cubes are
    already filtered at query time.
    """
    cube_file = event_dir / f"cube_{series}_{segment}.h5"
    if cube_file.exists():
        return cube_file
    pattern = f"*.{segment}.fits"
    exclude = None
    if series.startswith("hmi.") and email:
        files = sorted(glob.glob(str(event_dir / series / pattern)))
        if files:
            exclude = jsoc_bad_trecs(series, files, email)
    return build_cube(event_dir / series, cube_file, pattern=pattern,
                      exclude_trecs=exclude)


def load_cube(path: Path):
    """Return (data dataset [lazy], times as ISO strings, attrs dict)."""
    h5 = h5py.File(path, "r")
    return h5["data"], [t.decode() if isinstance(t, bytes) else t
                        for t in h5["t_obs"][:]], dict(h5.attrs)
