"""Remap tracked HMI cutouts onto the solar surface (events far from centre).

The pipeline measures on the image as seen. Within |lon| <= 40 deg that is
the accepted approximation; further out, foreshortening makes the spot
look smaller by mu and shortens east-west motions, so areas and moat
speeds "grow" as the region rotates in -- exactly the quantities whose
onsets we compare. For events flagged `deproject: true` the cube is built
on the solar surface instead:

  * grid: azimuthal-equidistant (Postel) projection centred, frame by
    frame, on the surface point under the im_patch centre, north up,
    at 0.362 Mm/px -- the pixel scale every downstream step assumes
    (PX_MM, HMI_PIXEL_KM), so spot areas, moat radii and FLCT speeds
    come out in true units with no further change;
  * interpolation: cubic spline (order 3); bilinear smoothing would vary
    with the sub-pixel offset from frame to frame and leak into LCT;
  * continuum only: divided by a per-frame quiet-Sun limb-darkening fit
    I_qs(mu) = a + b*mu, so quiet Sun = 1 across the patch. Across one
    cutout at lon -56 deg mu spans ~0.4-0.65 and limb darkening ~25 %,
    which a single per-frame median (penumbra.segment_frame) cannot absorb.
  * magnetograms stay line-of-sight (only the geometry changes); they
    serve as an MMF pattern tracer, where B_los/mu would only amplify
    noise toward the limb.

Geometry is computed analytically (heliographic <-> heliocentric <->
helioprojective, observer at DSUN_OBS, B0 = CRLT_OBS) and the last step to
CCD pixels goes through the frame's own FITS WCS, so CROTA2/CDELT/CRPIX
are honoured exactly. Verified against sunpy in tests/test_deproject.py.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

PX_MM = 0.362              # target pixel [Mm], = spottrack.PX_MM
N_TARGET = 496             # target grid; inside a 512 px patch at any mu
ARCSEC = np.pi / 180 / 3600


def _geometry(hdr):
    r = float(hdr.get("RSUN_REF", 696e6)) / 1e6           # Mm
    d = float(hdr["DSUN_OBS"]) / 1e6                       # Mm
    b0 = np.deg2rad(float(hdr["CRLT_OBS"]))
    return r, d, b0


def _wcs(hdr):
    keys = ("CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2",
            "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CROTA2")
    h = fits.Header()
    h["NAXIS"] = 2
    for k in keys:
        if k in hdr:
            h[k] = hdr[k]
    w = WCS(h, fix=False)
    w.wcs.set()
    return w


def hg_to_pixel(lon, lat, hdr, w=None):
    """Observer-relative heliographic lon/lat [rad] -> 0-based CCD pixel
    (x, y), plus mu (cosine of the heliocentric angle)."""
    r, d, b0 = _geometry(hdr)
    x = r * np.cos(lat) * np.sin(lon)
    y = r * (np.sin(lat) * np.cos(b0) - np.cos(lat) * np.cos(lon) * np.sin(b0))
    z = r * (np.sin(lat) * np.sin(b0) + np.cos(lat) * np.cos(lon) * np.cos(b0))
    dz = d - z
    dist = np.sqrt(x ** 2 + y ** 2 + dz ** 2)
    tx = np.arctan2(x, dz)
    ty = np.arcsin(y / dist)
    mu = (z * d - r ** 2) / (r * dist)
    w = w or _wcs(hdr)
    px, py = w.wcs_world2pix(np.rad2deg(tx), np.rad2deg(ty), 0)
    return px, py, mu


def pixel_to_hg(px, py, hdr, w=None):
    """0-based CCD pixel -> observer-relative heliographic lon/lat [rad]
    (NaN off-disk)."""
    r, d, b0 = _geometry(hdr)
    w = w or _wcs(hdr)
    tx, ty = (np.deg2rad(a) for a in w.wcs_pix2world(px, py, 0))
    dx, dy, dz = np.cos(ty) * np.sin(tx), np.sin(ty), -np.cos(ty) * np.cos(tx)
    od = d * dz                                  # observer at (0, 0, d)
    disc = od ** 2 - (d ** 2 - r ** 2)
    s = -od - np.sqrt(np.where(disc >= 0, disc, np.nan))
    x, y, z = s * dx, s * dy, d + s * dz
    lat = np.arcsin((y * np.cos(b0) + z * np.sin(b0)) / r)
    lon = np.arctan2(x, z * np.cos(b0) - y * np.sin(b0))
    return lon, lat


def postel_lonlat(lon_c, lat_c, n=N_TARGET, px_mm=PX_MM, r=696.0):
    """Heliographic lon/lat [rad] of an n x n Postel grid centred on
    (lon_c, lat_c); x = west, y = north, pixel (n-1)/2 at the centre."""
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n]
    X, Y = (xx - c) * px_mm, (yy - c) * px_mm
    rho_mm = np.hypot(X, Y)
    rho = rho_mm / r
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.cos(rho) * np.sin(lat_c)
                        + np.where(rho_mm > 0, Y * np.sin(rho) / rho_mm, 0)
                        * np.cos(lat_c))
        lon = lon_c + np.arctan2(X * np.sin(rho),
                                 rho_mm * np.cos(lat_c) * np.cos(rho)
                                 - Y * np.sin(lat_c) * np.sin(rho))
    return lon, lat


def patch_centre(hdr, shape, w=None):
    """Heliographic lon/lat [rad] under the centre of the cutout."""
    ny, nx = shape
    return pixel_to_hg((nx - 1) / 2, (ny - 1) / 2, hdr, w)


def limb_normalise(img, mu, spot_dilate_px=20, rough=0.9):
    """Divide by a quiet-Sun fit I_qs(mu) = a + b*mu (spots excluded)."""
    ok = np.isfinite(img) & np.isfinite(mu)
    med = np.nanmedian(img[ok])
    spot = ndimage.binary_dilation(ok & (img < rough * med),
                                   iterations=spot_dilate_px)
    sel = ok & ~spot
    if sel.sum() < 0.05 * ok.sum():
        # the rough mask swallowed the frame (very large spot or a bad
        # frame): undilated mask, then plain median as the last resort
        sel = ok & ~(img < rough * med)
        if sel.sum() < 0.05 * ok.sum():
            return img / med, (med, 0.0)
    a, b = 0.0, 0.0
    for _ in range(2):                      # one clipping pass
        A = np.c_[np.ones(sel.sum()), mu[sel]]
        a, b = np.linalg.lstsq(A, img[sel], rcond=None)[0]
        res = img - (a + b * mu)
        sig = 1.4826 * np.nanmedian(np.abs(res[sel]))
        sel &= np.abs(res) < 3 * sig
    return img / (a + b * mu), (a, b)


def deproject_frame(img, hdr, continuum, n=N_TARGET, px_mm=PX_MM):
    """One tracked cutout -> (n, n) surface map; returns (map, info)."""
    w = _wcs(hdr)
    lon_c, lat_c = patch_centre(hdr, img.shape, w)
    r = float(hdr.get("RSUN_REF", 696e6)) / 1e6
    lon, lat = postel_lonlat(lon_c, lat_c, n, px_mm, r)
    px, py, mu = hg_to_pixel(lon, lat, hdr, w)
    src = np.asarray(img, dtype=np.float64)
    fill = np.nanmedian(src)
    src = np.where(np.isfinite(src), src, fill)
    out = ndimage.map_coordinates(src, [py, px], order=3, mode="constant",
                                  cval=np.nan, prefilter=True)
    outside = (px < 0) | (py < 0) | (px > img.shape[1] - 1) | \
        (py > img.shape[0] - 1) | ~np.isfinite(px)
    out[outside] = np.nan
    coef = (np.nan, np.nan)
    if continuum:
        out, coef = limb_normalise(out, mu)
    n_bad = int((~np.isfinite(out)).sum())
    out = np.where(np.isfinite(out), out, 1.0 if continuum else 0.0)
    return out.astype("f4"), {
        "lon_c_deg": float(np.rad2deg(lon_c)),
        "lat_c_deg": float(np.rad2deg(lat_c)),
        "mu_c": float(mu[n // 2, n // 2]),
        "ld_a": float(coef[0]), "ld_b": float(coef[1]),
        "n_filled": n_bad}
