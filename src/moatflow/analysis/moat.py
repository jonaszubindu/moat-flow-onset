"""Moat-flow onset from FLCT flow maps.

The moat signature: an organized, azimuthally coherent radial outflow
(~0.5-1 km/s) in an annulus outside the (proto)spot boundary. Per flow
map we compute the azimuthally averaged radial velocity profile around
the spot barycenter; the onset metric is the mean outflow speed in the
annulus [r_spot + gap, r_spot + gap + width] vs time.
"""

import numpy as np

HMI_PIXEL_KM = 362.0


def spot_center(mask: np.ndarray) -> tuple[float, float]:
    """Barycenter (y, x) of the spot mask (umbra, or umbra+penumbra)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty spot mask")
    return float(ys.mean()), float(xs.mean())


def equivalent_radius_px(mask: np.ndarray) -> float:
    return float(np.sqrt(mask.sum() / np.pi))


def radial_profile(vx: np.ndarray, vy: np.ndarray, center: tuple[float, float],
                   r_max_px: float, dr_px: float = 2.0,
                   pixel_km: float = HMI_PIXEL_KM):
    """Azimuthal average of the radial velocity component.

    Returns (r_mm, v_r_mean, v_r_std) with r in Mm and v_r in km/s
    (positive = outflow).
    """
    ny, nx = vx.shape
    y, x = np.mgrid[0:ny, 0:nx]
    dy, dx = y - center[0], x - center[1]
    r = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_r = (vx * dx + vy * dy) / np.where(r > 0, r, np.nan)

    edges = np.arange(0, r_max_px, dr_px)
    r_mid, mean, std = [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        sel = (r >= e0) & (r < e1) & np.isfinite(v_r)
        if sel.sum() < 10:
            continue
        r_mid.append(0.5 * (e0 + e1) * pixel_km / 1e3)
        mean.append(float(np.nanmean(v_r[sel])))
        std.append(float(np.nanstd(v_r[sel])))
    return np.array(r_mid), np.array(mean), np.array(std)


def moat_outflow_series(vx_t: np.ndarray, vy_t: np.ndarray,
                        spot_masks: list[np.ndarray],
                        gap_mm: float = 1.0, width_mm: float = 6.0,
                        pixel_km: float = HMI_PIXEL_KM) -> np.ndarray:
    """Mean radial outflow [km/s] in the moat annulus, per time window.

    spot_masks: one spot mask per flow-map window (from the co-temporal
    continuum segmentation), so the annulus follows the growing spot.
    """
    out = np.full(len(vx_t), np.nan)
    px_mm = pixel_km / 1e3
    for k, (vx, vy, mask) in enumerate(zip(vx_t, vy_t, spot_masks)):
        if mask.sum() == 0:
            continue
        c = spot_center(mask)
        r_spot = equivalent_radius_px(mask)
        r0 = r_spot + gap_mm / px_mm
        r1 = r0 + width_mm / px_mm
        ny, nx = vx.shape
        y, x = np.mgrid[0:ny, 0:nx]
        dy, dx = y - c[0], x - c[1]
        r = np.hypot(dx, dy)
        sel = (r >= r0) & (r < r1)
        v_r = (vx[sel] * dx[sel] + vy[sel] * dy[sel]) / r[sel]
        out[k] = float(np.nanmean(v_r))
    return out
