"""Checks of moatflow.deproject on synthetic cutouts.

Run:  python tests/test_deproject.py      (pytest also collects it)

A spot and a granulation-like pattern are defined ON THE SPHERE and
rendered into a realistic HMI im_patch cutout (512 px, 0.504"/px) at
lon -56 deg, lat -15.5 deg -- AR13010's position at the start of its
window. Deprojection must return the true size, shape, flat quiet Sun and
true surface displacement; the forward geometry must agree with sunpy.
"""

import numpy as np

from moatflow.deproject import (PX_MM, deproject_frame, hg_to_pixel,
                                patch_centre, pixel_to_hg)

LON0, LAT0 = np.deg2rad(-56.0), np.deg2rad(-15.5)
R = 696.0


def make_header(lon=LON0, lat=LAT0, b0=-2.4, n=512, cdelt=0.504281104,
                dsun=151.5e9):
    """im_patch-like header with the patch centred on (lon, lat)."""
    hdr = {"CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN", "CUNIT1": "arcsec",
           "CUNIT2": "arcsec", "CRVAL1": 0.0, "CRVAL2": 0.0,
           "CDELT1": cdelt, "CDELT2": cdelt, "CROTA2": 0.0,
           "CRPIX1": 0.0, "CRPIX2": 0.0, "DSUN_OBS": dsun,
           "RSUN_REF": 696e6, "CRLT_OBS": b0}
    # place the centre: find where (lon, lat) falls with CRPIX = 0, then
    # shift CRPIX so it lands on the patch centre pixel
    px, py, _ = hg_to_pixel(np.array(lon), np.array(lat), hdr)
    hdr["CRPIX1"] = (n - 1) / 2 - float(px)
    hdr["CRPIX2"] = (n - 1) / 2 - float(py)
    return hdr


def postel_xy(lon, lat, lon_c=LON0, lat_c=LAT0):
    """Forward azimuthal-equidistant projection [Mm]."""
    cosc = (np.sin(lat_c) * np.sin(lat)
            + np.cos(lat_c) * np.cos(lat) * np.cos(lon - lon_c))
    c = np.arccos(np.clip(cosc, -1, 1))
    k = np.where(c > 0, c / np.sin(np.where(c > 0, c, 1)), 1.0)
    x = R * k * np.cos(lat) * np.sin(lon - lon_c)
    y = R * k * (np.cos(lat_c) * np.sin(lat)
                 - np.sin(lat_c) * np.cos(lat) * np.cos(lon - lon_c))
    return x, y


def render(hdr, surface_fn, n=512, limb=True):
    """CCD image of a function of surface (Postel) coordinates [Mm]."""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    lon, lat = pixel_to_hg(xx, yy, hdr)
    X, Y = postel_xy(lon, lat)
    _, _, mu = hg_to_pixel(lon, lat, hdr)
    img = surface_fn(X, Y)
    if limb:
        img = img * (1 - 0.6 * (1 - mu))        # linear limb darkening
    return img


def test_matches_sunpy():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    hdr = make_header()
    obs = SkyCoord(0 * u.deg, hdr["CRLT_OBS"] * u.deg, hdr["DSUN_OBS"] * u.m,
                   frame=frames.HeliographicStonyhurst,
                   obstime="2022-05-14T20:30")
    lon = np.deg2rad(np.array([-60.0, -56.0, -50.0, -20.0, 10.0]))
    lat = np.deg2rad(np.array([-20.0, -15.5, -10.0, 5.0, 30.0]))
    c = SkyCoord(np.rad2deg(lon) * u.deg, np.rad2deg(lat) * u.deg,
                 frame=frames.HeliographicStonyhurst, obstime=obs.obstime,
                 rsun=696 * u.Mm, observer=obs)
    hpc = c.transform_to(frames.Helioprojective(observer=obs,
                                                obstime=obs.obstime))
    # our CCD pixel -> back to HPC through the same TAN WCS
    from moatflow.deproject import _wcs
    px, py, _ = hg_to_pixel(lon, lat, hdr)
    tx, ty = _wcs(hdr).wcs_pix2world(px, py, 0)
    tx = (tx + 180) % 360 - 180            # WCS returns [0, 360)
    err = np.hypot(tx * 3600 - hpc.Tx.to_value(u.arcsec),
                   ty * 3600 - hpc.Ty.to_value(u.arcsec))
    assert err.max() < 0.01, err          # arcsec
    # round trip pixel -> hg -> pixel
    lon2, lat2 = pixel_to_hg(px, py, hdr)
    assert np.allclose(lon2, lon, atol=1e-9) and np.allclose(lat2, lat,
                                                             atol=1e-9)
    print(f"sunpy agreement: max {err.max() * 1000:.2f} mas; round trip ok")


def test_centre():
    hdr = make_header()
    lon_c, lat_c = patch_centre(hdr, (512, 512))
    assert abs(lon_c - LON0) < 1e-9 and abs(lat_c - LAT0) < 1e-9


def test_spot_true_size_and_flat_quiet_sun():
    hdr = make_header()
    r_spot = 10.0                                # Mm on the surface
    img = render(hdr, lambda X, Y: np.where(np.hypot(X, Y) < r_spot, 0.3,
                                            1.0))
    _, _, mu = hg_to_pixel(LON0, LAT0, hdr)
    # as seen, geometry only (with limb darkening a global threshold also
    # catches the dark limb-side corner -- the reason for limb_normalise)
    flat = render(hdr, lambda X, Y: np.where(np.hypot(X, Y) < r_spot, 0.3,
                                             1.0), limb=False)
    ccd_area = (flat < 0.65).sum() * 0.3646 ** 2
    dp, info = deproject_frame(img, hdr, continuum=True)
    spot = dp < 0.65
    area = spot.sum() * PX_MM ** 2
    true = np.pi * r_spot ** 2
    yy, xx = np.nonzero(spot)
    ax = np.sqrt(np.linalg.eigvalsh(np.cov(np.c_[xx, yy].T)))
    qs = dp[~ndilate(spot, 25)]
    print(f"spot at mu={float(mu):.2f}: true {true:.0f} Mm^2, CCD as seen "
          f"{ccd_area:.0f}, deprojected {area:.0f}; axis ratio "
          f"{ax.min() / ax.max():.3f}; quiet Sun {qs.mean():.3f} "
          f"+- {qs.std():.4f}; filled px {info['n_filled']}")
    assert abs(area / true - 1) < 0.03
    assert ax.min() / ax.max() > 0.97
    assert abs(qs.mean() - 1) < 0.005 and qs.std() < 0.01
    assert info["n_filled"] == 0


def ndilate(m, it):
    from scipy import ndimage
    return ndimage.binary_dilation(m, iterations=it)


def test_surface_shift_recovered():
    """A pattern moved 1 Mm west on the surface moves 1/PX_MM px in the
    deprojected map (on the CCD only ~mu of that)."""
    hdr = make_header()
    rng = np.random.default_rng(1)
    k = rng.uniform(0.3, 1.2, (40, 2)) * rng.choice([-1, 1], (40, 2))
    ph = rng.uniform(0, 2 * np.pi, 40)

    def pattern(dx):
        return lambda X, Y: 1 + 0.05 * sum(
            np.cos(k[i, 0] * (X - dx) + k[i, 1] * Y + ph[i])
            for i in range(40)) / np.sqrt(40)
    a, _ = deproject_frame(render(hdr, pattern(0.0)), hdr, continuum=True)
    b, _ = deproject_frame(render(hdr, pattern(1.0)), hdr, continuum=True)
    sx, sy = phase_shift(a, b)
    ca, cb = render(hdr, pattern(0.0), limb=False), \
        render(hdr, pattern(1.0), limb=False)
    cx, _ = phase_shift(*(np.nan_to_num(c, nan=1.0)[128:384, 128:384]
                          for c in (ca, cb)))
    print(f"1 Mm west on the surface: deprojected {sx * PX_MM:.3f} Mm "
          f"(y {sy * PX_MM:+.3f}); CCD as seen {cx * 0.3646:.3f} Mm")
    assert abs(sx * PX_MM - 1.0) < 0.02 and abs(sy * PX_MM) < 0.02


def phase_shift(a, b):
    """Sub-pixel shift of b relative to a (x, y) by cross-correlation."""
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2((a - a.mean()) * w), np.fft.fft2((b - b.mean()) * w)
    cc = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(A) * B)))
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)

    def sub(c_m, c_0, c_p):
        return 0.5 * (c_m - c_p) / (c_m - 2 * c_0 + c_p)
    dx = ix + sub(cc[iy, ix - 1], cc[iy, ix], cc[iy, ix + 1])
    dy = iy + sub(cc[iy - 1, ix], cc[iy, ix], cc[iy + 1, ix])
    return dx - a.shape[1] // 2, dy - a.shape[0] // 2


if __name__ == "__main__":
    test_matches_sunpy()
    test_centre()
    test_spot_true_size_and_flat_quiet_sun()
    test_surface_shift_recovered()
    print("all deprojection checks passed")
