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


def build_cube(fits_dir: Path, out_file: Path,
               pattern: str = "*.fits") -> Path:
    files = sorted(glob.glob(str(fits_dir / pattern)))
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


def load_cube(path: Path):
    """Return (data dataset [lazy], times as ISO strings, attrs dict)."""
    h5 = h5py.File(path, "r")
    return h5["data"], [t.decode() if isinstance(t, bytes) else t
                        for t in h5["t_obs"][:]], dict(h5.attrs)
