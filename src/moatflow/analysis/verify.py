"""Physical verification of the FLCT flows, run per event by onset_analysis.

The three checks the LCT literature uses (VETTING.md section H):

1. Divergence map at a mature epoch: supergranular cells plus the moat
   as a divergence ring around the tracked spot. Tracking noise has no
   such organisation.
2. Doppler cross-check: horizontal flows have a line-of-sight component
   away from disk centre. The azimuthal pattern of the FLCT flows
   projected onto the LOS must match the Dopplergram in the moat
   annulus. Needs the hmi.V_45s cube; skipped without it.
3. Shrinking-Sun immunity: the known LCT artefact (Loptien et al. 2016)
   is a large-scale, near-uniform apparent flow toward disk centre. It
   must cancel in the azimuthal mean of the radial component; this
   removes the patch-mean flow and shows the moat curve does not move.

Nothing is tuned per event by hand. The AR's heliographic position comes
from the catalog (lon_ref/lat_ref at t_ref, carried forward with the
photospheric rotation rate); the divergence epoch is the tracking seed
(a mature spot); the Doppler epoch is the post-formation epoch where a
horizontal flow has the largest LOS component. Numbers are written to
quicklook/verify_flct.json for the sample table.
"""

import json
from datetime import timedelta

import numpy as np
from scipy import ndimage

from ..viz import parse_datetimes
from .audit import radial_velocity
from .spottrack import PX_MM

KM_PER_PX = 362.0


def synodic_rate_deg_per_h(lat_deg):
    """Photospheric differential rotation (Snodgrass & Ulrich 1990), synodic."""
    s2 = np.sin(np.deg2rad(lat_deg)) ** 2
    sidereal = 14.713 - 2.396 * s2 - 1.787 * s2 ** 2          # deg/day
    return (sidereal - 0.9856) / 24.0


def los_coefficients(lon_deg, lat_deg, b0_deg):
    """(a, b) with v_away = a*vx + b*vy for a horizontal flow at (lon, lat).

    vx = solar west, vy = solar north, v_away positive = redshift.
    Observer at heliographic latitude B0.
    """
    l, b, B0 = np.deg2rad(lon_deg), np.deg2rad(lat_deg), np.deg2rad(b0_deg)
    return (np.cos(B0) * np.sin(l),
            np.cos(B0) * np.sin(b) * np.cos(l) - np.sin(B0) * np.cos(b))


def _b0(when):
    try:
        from astropy.time import Time
        from sunpy.coordinates import sun
        return float(sun.B0(Time(when)).deg)
    except Exception:
        return 0.0


def pick_epochs(t_h, usable, pf_lit, lon, lat, b0, seed_h, margin_h=1.0):
    """(k_div, k_dop): mature epoch, and max-LOS-sensitivity epoch after
    penumbra formation (falls back to any usable epoch)."""
    ok = usable & (t_h >= t_h[0] + margin_h) & (t_h <= t_h[-1] - margin_h)
    if not ok.any():
        ok = usable.copy()
    idx = np.flatnonzero(ok)
    k_div = int(idx[np.argmin(np.abs(t_h[idx] - seed_h))])
    after = pf_lit[1] if np.isfinite(pf_lit[1]) else pf_lit[0]
    post = ok & (t_h >= after) if np.isfinite(after) else ok
    if not post.any():
        post = ok
    a, b = los_coefficients(lon, lat, b0)
    sens = np.where(post, np.hypot(a, b), -1.0)
    return k_div, int(np.argmax(sens))


def _annulus_circles(ax, tr, k, aud):
    from matplotlib.patches import Circle
    cy, cx = tr["center"][k]
    for r in (aud["r0_mm"][k], aud["r1_mm"][k]):
        ax.add_patch(Circle((cx, cy), r / PX_MM, fill=False, color="green",
                            ls="--", lw=1.0))
    ax.plot(cx, cy, "g+", ms=11)


def divergence_check(frame, VX, VY, k, tr, aud, png, event_id, t_h):
    import matplotlib.pyplot as plt
    sl = slice(max(k - 1, 0), k + 2)                 # 3 h average
    vx, vy = np.nanmean(VX[sl], 0), np.nanmean(VY[sl], 0)
    div = (np.gradient(vx, KM_PER_PX, axis=1)
           + np.gradient(vy, KM_PER_PX, axis=0))
    div_s = ndimage.gaussian_filter(np.nan_to_num(div), 8)
    ann, bad = aud["cache"][k][2], aud["cache"][k][6]
    sel = ann & ~bad
    d_ann = float(np.mean(div_s[sel]) * 1e5)
    frac_pos = float(np.mean(div_s[sel] > 0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(frame, cmap="afmhot", origin="lower",
                   vmin=np.percentile(frame, 0.5),
                   vmax=np.percentile(frame, 99.9))
    axes[0].set_title(f"continuum, t = {t_h[k]:.0f} h")
    im = axes[1].imshow(div_s * 1e5, cmap="RdBu_r", origin="lower",
                        vmin=-1.5, vmax=1.5)
    axes[1].set_title(f"FLCT divergence [$10^{{-5}}$ s$^{{-1}}$], 3 h avg\n"
                      f"moat annulus mean {d_ann:+.2f}, "
                      f"{frac_pos*100:.0f}% of it diverging")
    plt.colorbar(im, ax=axes[1], shrink=0.8)
    for ax in axes:
        _annulus_circles(ax, tr, k, aud)
        ax.set_axis_off()
    fig.suptitle(f"{event_id} — divergence check", y=0.99)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return {"epoch_h": float(t_h[k]), "annulus_mean_div_1e-5_per_s": d_ann,
            "annulus_frac_diverging": frac_pos}


def doppler_check(vfile, VX, VY, t_mid, t0, k, tr, aud, lon_k, lat, b0,
                  png, event_id):
    import matplotlib.pyplot as plt
    from ..cubes import load_cube
    vc, v_iso, _ = load_cube(vfile)
    v_s = np.array([(d - t0).total_seconds() for d in parse_datetimes(v_iso)])
    sel_frames = np.flatnonzero(np.abs(v_s - t_mid[k]) <= 3600)[::4]
    if len(sel_frames) < 10:
        print(f"  Doppler check skipped: only {len(sel_frames)} "
              "Dopplergram frames within +-1 h of the epoch")
        return None
    dop = np.nanmean([np.asarray(vc[i]) for i in sel_frames], axis=0)

    # background (rotation gradient, convective blueshift, meridional
    # terms): smooth with the spot filled in, so its strong Evershed
    # signal does not leak into the background under the annulus
    cy, cx = tr["center"][k]
    spot = ndimage.binary_dilation(tr["masks"][k], iterations=5)
    src = np.nan_to_num(dop, nan=np.nanmedian(dop))
    src[spot] = np.median(src[~spot])
    dop_res = (dop - ndimage.gaussian_filter(src, 60)) / 1e3      # km/s

    sl = slice(max(k - 1, 0), k + 2)
    vx, vy = np.nanmean(VX[sl], 0), np.nanmean(VY[sl], 0)
    a, b = los_coefficients(lon_k, lat, b0)
    pred = a * vx + b * vy

    rr, phi, ann = aud["cache"][k][0], aud["cache"][k][1], aud["cache"][k][2]
    bad = aud["cache"][k][6]
    bins = np.linspace(-np.pi, np.pi, 19)
    phic = 0.5 * (bins[:-1] + bins[1:])
    pd_, pp_ = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = ann & ~bad & (phi >= lo) & (phi < hi)
        pd_.append(np.nanmean(dop_res[s]) if s.sum() > 10 else np.nan)
        pp_.append(np.nanmean(pred[s]) if s.sum() > 10 else np.nan)
    pd_, pp_ = np.array(pd_), np.array(pp_)
    g = np.isfinite(pd_) & np.isfinite(pp_)
    r = float(np.corrcoef(pd_[g], pp_[g])[0, 1])
    ratio = float(np.sum(pp_[g] * pd_[g]) / np.sum(pd_[g] ** 2))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.rad2deg(phic), pd_, "o-",
            label="Dopplergram (2 h avg, background removed)")
    ax.plot(np.rad2deg(phic), pp_, "s-", label="FLCT flows projected on LOS")
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="azimuth around spot [deg] (0 = solar W)",
           ylabel="LOS velocity in moat annulus [km/s]",
           title=f"{event_id} Doppler cross-check, t = {t_mid[k]/3600:.0f} h, "
                 f"AR at lon {lon_k:+.0f}, lat {lat:+.0f} deg\n"
                 f"r = {r:.2f},  FLCT/Doppler amplitude = {ratio:.2f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return {"epoch_h": float(t_mid[k] / 3600), "lon_deg": float(lon_k),
            "r": r, "flct_over_doppler": ratio}


def shrinking_sun_check(VX, VY, tr, aud, t_h, png, event_id):
    import matplotlib.pyplot as plt
    mvx = np.nanmean(VX, axis=(1, 2))
    mvy = np.nanmean(VY, axis=(1, 2))
    v_cor = np.full(len(t_h), np.nan)
    for k, c in aud["cache"].items():
        rr, ann, bad = c[0], c[2], c[6]
        cy, cx = tr["center"][k]
        vr = radial_velocity(VX[k] - mvx[k], VY[k] - mvy[k], cy, cx, rr)
        s = ann & ~bad & np.isfinite(vr)
        if s.sum() > 30:
            v_cor[k] = np.nanmean(vr[s])
    dif = v_cor - aud["v_clean"]
    g = np.isfinite(dif)
    q = len(t_h) // 4
    drift = float((np.nanmean(mvx[-q:]) - np.nanmean(mvx[:q])) * 1000)
    shift = float(np.nanmax(np.abs(dif[g])) * 1000)
    corr = float(np.corrcoef(v_cor[g], aud["v_clean"][g])[0, 1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t_h, mvx * 1000, label="<v$_x$> over patch")
    ax[0].plot(t_h, mvy * 1000, label="<v$_y$> over patch")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(xlabel="hours since start", ylabel="patch-mean flow [m/s]",
              title="the shrinking-Sun artefact itself\n"
                    "(converges on disk centre, reverses at the meridian)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(t_h, aud["v_clean"], lw=1.4, label="moat curve (quoted)")
    ax[1].plot(t_h, v_cor, "--", lw=1.2, label="patch-mean flow removed")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="hours since start", ylabel="v$_r$ [km/s]",
              title=f"immunity: max shift {shift:.1f} m/s, r = {corr:.4f}")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"{event_id} — shrinking-Sun check", y=1.0)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return {"patch_mean_vx_drift_m_s": drift,
            "max_patch_mean_speed_m_s": float(np.nanmax(np.hypot(mvx, mvy)) * 1000),
            "max_curve_shift_m_s": shift, "curve_correlation": corr}


def run_verification(out_dir, event_id, event, frames, tr, VX, VY, t_mid,
                     t0, aud, seed_h, pf_lit):
    """Run all three checks; write PNGs + verify_flct.json; return results."""
    ql = out_dir / "quicklook"
    ql.mkdir(exist_ok=True)
    t_h = t_mid / 3600
    lat = float(event["lat_ref"])
    t_ref_h = (parse_datetimes([event["t_ref"]])[0] - t0).total_seconds() / 3600
    lon = float(event["lon_ref"]) + synodic_rate_deg_per_h(lat) * (t_h - t_ref_h)
    b0 = _b0(t0 + timedelta(hours=float(t_h[len(t_h) // 2])))
    usable = np.array([k in aud["cache"] for k in range(len(t_h))])
    if not usable.any():
        print("  verification skipped: no usable (tracked) epochs")
        return None

    k_div, k_dop = pick_epochs(t_h, usable, pf_lit, lon, lat, b0, seed_h)
    res = {"event": event_id, "lat_deg": lat, "b0_deg": b0,
           "lon_range_deg": [float(lon[0]), float(lon[-1])]}
    res["divergence"] = divergence_check(frames[k_div], VX, VY, k_div, tr,
                                         aud, ql / "flct_divergence.png",
                                         event_id, t_h)
    vfile = out_dir / "cube_hmi.V_45s_Dopplergram.h5"
    if vfile.exists():
        res["doppler"] = doppler_check(vfile, VX, VY, t_mid, t0, k_dop, tr,
                                       aud, float(lon[k_dop]), lat, b0,
                                       ql / "flct_doppler_check.png", event_id)
    else:
        res["doppler"] = None
        print("  Doppler check skipped: no cube_hmi.V_45s_Dopplergram.h5 "
              "(download hmi.V_45s + run quicklook.py on it to enable)")
    res["shrinking_sun"] = shrinking_sun_check(VX, VY, tr, aud, t_h,
                                               ql / "flct_shrinking_sun.png",
                                               event_id)
    with open(ql / "verify_flct.json", "w") as f:
        json.dump(res, f, indent=2)
    return res
