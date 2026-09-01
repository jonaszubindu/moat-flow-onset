"""FLCT (Fisher & Welsch 2008) horizontal velocities from image cubes.

Standard recipe for HMI moat-flow studies:
  * pairwise FLCT at the native cadence (45 s continuum, or 720 s Blos),
  * average the pairwise flow maps over 1-2 h windows to suppress
    granular/oscillatory noise and bring out the organized moat outflow,
  * Gaussian apodization sigma of ~1-2 Mm (2.8-5.6 HMI pixels of 362 km).
"""

import numpy as np

try:
    import pyflct
except ImportError as e:  # pragma: no cover
    pyflct = None
    _import_error = e

HMI_PIXEL_KM = 362.0  # 0.504 arcsec at disk center


def flct_pair(im1: np.ndarray, im2: np.ndarray, dt_s: float,
              sigma_px: float = 5.0, pixel_km: float = HMI_PIXEL_KM,
              kr: float | None = 0.5, thresh: float | None = None):
    """Velocity maps (vx, vy in km/s, plus mask) between two frames.

    kr applies FLCT's low-pass filter (fraction of Nyquist) — recommended
    for noisy magnetograms; set kr=None to disable.
    thresh skips pixels where the mean |image| is below it (image units,
    e.g. Gauss) — essential for magnetograms, where sparse MMFs would
    otherwise be averaged away against noise pixels that FLCT reports as
    zero velocity. Skipped pixels return vm=0; mask them before averaging.
    """
    if pyflct is None:
        raise ImportError("pyflct is required for FLCT tracking") from _import_error
    kwargs = {} if kr is None else {"kr": kr}
    if thresh is not None:
        kwargs["thresh"] = thresh
    vx, vy, vm = pyflct.flct(im1.astype("f8"), im2.astype("f8"),
                             dt_s, pixel_km, sigma_px, quiet=True, **kwargs)
    return vx, vy, vm


def flct_windowed(cube, times_s: np.ndarray, window_s: float = 3600.0,
                  pair_stride: int = 1, sigma_px: float = 5.0,
                  pixel_km: float = HMI_PIXEL_KM, kr: float | None = 0.5,
                  thresh: float | None = None):
    """Time-averaged flow maps over consecutive windows.

    Parameters
    ----------
    cube : (nt, ny, nx) array-like (h5py dataset works, frames read lazily)
    times_s : observation times in seconds (monotonic)
    window_s : averaging window (1-2 h typical)
    pair_stride : frame separation of each FLCT pair (1 = native cadence)

    Returns
    -------
    t_mid : (nw,) window mid-times [s]
    vx, vy : (nw, ny, nx) window-averaged velocities [km/s]
    """
    nt = len(times_s)
    edges = np.arange(times_s[0], times_s[-1], window_s)
    t_mid, vx_out, vy_out = [], [], []

    for e0 in edges:
        idx = np.flatnonzero((times_s >= e0) & (times_s < e0 + window_s))
        pairs = [(i, i + pair_stride) for i in idx
                 if i + pair_stride < nt and times_s[i + pair_stride] < e0 + window_s]
        if not pairs:
            continue
        acc_x = acc_y = cnt = None
        for i, j in pairs:
            dt = float(times_s[j] - times_s[i])
            vx, vy, vm = flct_pair(cube[i], cube[j], dt, sigma_px, pixel_km,
                                   kr, thresh)
            # with thresh, only pixels FLCT actually computed (vm>0) enter
            # the average — sparse MMFs would otherwise be diluted by the
            # zero velocities FLCT reports for skipped noise pixels
            good = (vm > 0) if thresh is not None else np.ones(vx.shape, bool)
            if acc_x is None:
                acc_x = np.zeros(vx.shape)
                acc_y = np.zeros(vx.shape)
                cnt = np.zeros(vx.shape)
            acc_x[good] += vx[good]
            acc_y[good] += vy[good]
            cnt[good] += 1
        t_mid.append(e0 + window_s / 2)
        with np.errstate(invalid="ignore"):
            vx_out.append(np.where(cnt > 0, acc_x / np.maximum(cnt, 1), np.nan))
            vy_out.append(np.where(cnt > 0, acc_y / np.maximum(cnt, 1), np.nan))

    return np.array(t_mid), np.array(vx_out), np.array(vy_out)
