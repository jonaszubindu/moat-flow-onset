"""Detect FITS files that arrived truncated.

drms reports a file as downloaded as soon as it has written something,
so a transfer cut short mid-file still counts. Counting files therefore
cannot tell a complete chunk from a damaged one — only opening them can.
A truncated file then surfaces much later, when the cube is built.
"""

import os
import warnings


def is_truncated(path) -> bool:
    """True if the file cannot be read as a complete FITS image."""
    from astropy.io import fits
    try:
        with warnings.catch_warnings():
            # astropy warns rather than raises on a short file; the data
            # read below would fail anyway, but promoting the warning
            # catches the case where only the heap is missing.
            warnings.filterwarnings("error", message=".*truncated.*")
            with fits.open(path) as hdul:
                hdu = next(h for h in hdul if h.data is not None)
                _ = hdu.data[0, 0]
        return False
    except Exception:
        return True


def bad_fits(paths, verbose=False) -> list:
    """Subset of `paths` that is truncated or otherwise unreadable."""
    bad = []
    for p in paths:
        if is_truncated(p):
            bad.append(p)
            if verbose:
                print(f"    truncated: {os.path.basename(p)} "
                      f"({os.path.getsize(p)} bytes)")
    return bad
