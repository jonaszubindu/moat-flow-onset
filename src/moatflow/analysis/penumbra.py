"""Penumbra onset from continuum intensity segmentation.

Classic thresholds relative to the quiet-sun intensity I_qs:
    umbra     I < 0.55 I_qs
    penumbra  0.55 I_qs <= I < 0.90 I_qs
(Thresholds are parameters — calibrate on a few frames per event; limb
darkening is negligible within a tracked patch at |lon| <= 40 deg but the
quiet-sun normalization is computed per frame to absorb it.)
"""

import numpy as np
from scipy import ndimage

HMI_PIXEL_KM = 362.0


def quiet_sun_intensity(frame: np.ndarray, spot_dilate_px: int = 20,
                        rough_thresh: float = 0.9) -> float:
    """Median intensity outside a dilated rough spot mask."""
    med = np.nanmedian(frame)
    rough = frame < rough_thresh * med
    excl = ndimage.binary_dilation(rough, iterations=spot_dilate_px)
    return float(np.nanmedian(frame[~excl]))


def segment_frame(frame: np.ndarray, umbra_thresh: float = 0.55,
                  penumbra_thresh: float = 0.90,
                  min_feature_px: int = 20):
    """Return (umbra_mask, penumbra_mask), small speckles removed."""
    iqs = quiet_sun_intensity(frame)
    umbra = frame < umbra_thresh * iqs
    penumbra = (frame >= umbra_thresh * iqs) & (frame < penumbra_thresh * iqs)
    # keep only penumbra touching an umbra/pore, drop isolated darkenings
    labels, _ = ndimage.label(penumbra | umbra)
    keep = np.unique(labels[umbra])
    penumbra &= np.isin(labels, keep[keep > 0])
    for mask in (umbra, penumbra):
        lab, n = ndimage.label(mask)
        sizes = ndimage.sum_labels(mask, lab, index=np.arange(1, n + 1))
        mask &= np.isin(lab, 1 + np.flatnonzero(sizes >= min_feature_px))
    return umbra, penumbra


def area_series(cube, pixel_km: float = HMI_PIXEL_KM, **kwargs):
    """Umbral and penumbral areas [Mm^2] vs frame index."""
    a_u, a_p = [], []
    px_mm2 = (pixel_km / 1e3) ** 2
    for i in range(len(cube)):
        u, p = segment_frame(np.asarray(cube[i]), **kwargs)
        a_u.append(u.sum() * px_mm2)
        a_p.append(p.sum() * px_mm2)
    return np.array(a_u), np.array(a_p)
