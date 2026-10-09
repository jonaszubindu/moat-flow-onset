"""Read SSTRED narrowband cubes (CRISP / CHROMIS ``nb_*_im.fits``).

Layout (SSTRED, checked on a real cube):
  primary HDU  float32, numpy shape (nscan, nstokes, ntune, ny, nx);
               BUNIT W m^-2 Hz^-1 sr^-1 (absolute), CTYPE3 'WAVE-TAB' [nm],
               CTYPE5 'UTC--TAB' [s since DATEREF], CTYPE1/2 'HPLN/HPLT-TAB'
  WCS-TAB      column 'HPLN+HPLT+WAVE+TIME', numpy shape
               (nscan, ntune, 2, 2, 4): HPLN, HPLT [arcsec], WAVE [nm],
               TIME [s] at the field corners (HPLN-INDEX/HPLT-INDEX give
               the corner pixels, 1-based)

The primary HDU is memory-mapped, so a scan is read without loading the
whole time series.
"""

from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits


class SSTRedCube:
    def __init__(self, path):
        self.path = str(path)
        self.hdul = fits.open(self.path, memmap=True)
        self.data = self.hdul[0].data                 # (nscan, ns, nw, ny, nx)
        self.header = self.hdul[0].header
        if self.data.ndim != 5:
            raise ValueError(f"{self.path}: expected a 5-D SSTRED cube, "
                             f"got shape {self.data.shape}")
        self.nscan, self.nstokes, self.ntune, self.ny, self.nx = self.data.shape
        tab = self.hdul["WCS-TAB"].data
        self.coords = np.asarray(tab["HPLN+HPLT+WAVE+TIME"][0], dtype=float)
        self.corner_px = (np.asarray(tab["HPLN-INDEX"][0]) - 1,
                          np.asarray(tab["HPLT-INDEX"][0]) - 1)
        ref = self.header.get("DATEREF", self.header.get("DATE-BEG", "")[:10]
                              + "T00:00:00")
        self.dateref = datetime.fromisoformat(str(ref)[:26])

    # -- coordinates --------------------------------------------------------
    def wavelengths_A(self, scan=0):
        """Tuning wavelengths of one scan [Angstrom]."""
        return self.coords[scan, :, 0, 0, 2] * 10.0

    def time(self, scan):
        return self.dateref + timedelta(seconds=float(self.coords[scan, 0, 0, 0, 3]))

    def pointing(self, scan=0):
        """(x_centre, y_centre, plate scale) in arcsec, from the corners."""
        c = self.coords[scan, 0]                        # (2, 2, 4)
        x = c[:, :, 0].mean()
        y = c[:, :, 1].mean()
        (i0, i1), (j0, j1) = self.corner_px
        sx = abs(c[0, 1, 0] - c[0, 0, 0]) / max(i1 - i0, 1)
        sy = abs(c[1, 0, 1] - c[0, 0, 1]) / max(j1 - j0, 1)
        return float(x), float(y), float(0.5 * (sx + sy))

    def mu(self, scan=0):
        """cos(heliocentric angle) at the field centre."""
        x, y, _ = self.pointing(scan)
        r = np.hypot(x, y) / solar_radius_arcsec(self.time(scan))
        return float(np.sqrt(max(1.0 - r * r, 0.0)))

    # -- data ---------------------------------------------------------------
    def scan(self, k, crop=None, binning=1):
        """Stokes cube of scan k as (ny, nx, 4, ntune) float32.

        crop    : (x0, x1, y0, y1) pixel box, applied before binning
        binning : average binning x binning pixels
        """
        d = self.data[k]                                # (ns, nw, ny, nx)
        if crop is not None:
            x0, x1, y0, y1 = crop
            d = d[:, :, y0:y1, x0:x1]
        d = np.asarray(d, dtype=np.float32)
        if binning > 1:
            ns, nw, ny, nx = d.shape
            ny, nx = ny // binning * binning, nx // binning * binning
            d = d[:, :, :ny, :nx].reshape(ns, nw, ny // binning, binning,
                                          nx // binning, binning).mean(axis=(3, 5))
        return np.ascontiguousarray(d.transpose(2, 3, 0, 1))

    def close(self):
        self.hdul.close()


def solar_radius_arcsec(when):
    """Apparent solar radius [arcsec] (Earth-Sun distance from the orbit's
    eccentricity; good to ~0.1 %, plenty for mu)."""
    doy = when.timetuple().tm_yday + when.hour / 24.0
    d_au = 1.0 - 0.01672 * np.cos(2 * np.pi * (doy - 4.0) / 365.256)
    return 959.63 / d_au
