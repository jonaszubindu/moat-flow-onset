"""Audit of what enters the moat-outflow average, and its verification movie.

The onset curve is an azimuthal mean of the radial velocity over an
annulus that follows the tracked spot. That mean can be produced by
things other than a moat: a couple of hot sectors, a neighbouring pore
inside the annulus, or an annulus that sits beside the flow peak rather
than on it. Everything here exists to make those failure modes visible
and quantified rather than assumed.

Annulus geometry, two conventions (see `annulus_geometry`):
  "fixed"  inner edge r_spot + gap_mm, width width_mm   [default]
  "scaled" inner edge f0 * r_spot, outer edge f1 * r_spot
The moat extends ~1-2 spot radii beyond the spot boundary, so a fixed
window samples a shrinking fraction of the moat as the spot grows; the
scaled window keeps the relative position constant but grows into the
neighbours. `scripts/annulus_sensitivity.py` compares them per event.
"""

import matplotlib
import numpy as np
from scipy import ndimage

from .penumbra import segment_frame
from .spottrack import PX_MM

GAP_MM, WIDTH_MM = 1.0, 6.0
SCALE_F0, SCALE_F1 = 1.2, 2.5
N_SECT = 12
SECT_REJECT = 0.35


def annulus_geometry(shape, cy, cx, r_spot_px, mode="fixed",
                     gap_mm=GAP_MM, width_mm=WIDTH_MM,
                     f0=SCALE_F0, f1=SCALE_F1):
    """(rr, phi, annulus mask, r0, r1) in pixels."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    if mode == "fixed":
        r0 = r_spot_px + gap_mm / PX_MM
        r1 = r0 + width_mm / PX_MM
    elif mode == "scaled":
        r0, r1 = f0 * r_spot_px, f1 * r_spot_px
    else:
        raise ValueError("mode must be 'fixed' or 'scaled'")
    ann = (rr >= r0) & (rr < r1)
    phi = np.arctan2(yy - cy, xx - cx)
    return rr, phi, ann, r0, r1


def contamination_mask(frame, own_mask, dilate=3):
    """Spot/pore pixels that are NOT the tracked spot, dilated.

    Dilated because granulation adjacent to a dark feature is disturbed,
    so its LCT signal is not moat flow either.
    """
    u, p = segment_frame(frame)
    others = (u | p) & ~ndimage.binary_dilation(own_mask, iterations=2)
    return ndimage.binary_dilation(others, iterations=dilate)


def radial_velocity(vx, vy, cy, cx, rr):
    with np.errstate(invalid="ignore"):
        return ((vx * (np.arange(vx.shape[1])[None, :] - cx)
                 + vy * (np.arange(vx.shape[0])[:, None] - cy))
                / np.where(rr > 0, rr, np.nan))


def sector_profile(vr, ann, phi, bad):
    """(sector centres, mean v_r, rejected flag)."""
    edges = np.linspace(-np.pi, np.pi, N_SECT + 1)
    mid, vals, rejected = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sec = ann & (phi >= a) & (phi < b)
        if sec.sum() == 0:
            continue
        good = sec & ~bad & np.isfinite(vr)
        mid.append(0.5 * (a + b))
        vals.append(np.nanmean(vr[good]) if good.sum() > 10 else np.nan)
        rejected.append((sec & bad).sum() / sec.sum() > SECT_REJECT)
    return np.array(mid), np.array(vals), np.array(rejected)


def audit_series(frames, tr, VX, VY, mode="fixed", **geom):
    """Per-epoch audit quantities and cached per-frame geometry.

    Returns dict with v_raw (all annulus pixels), v_clean (contamination
    masked — the quoted curve), frac_contaminated, n_sectors_rejected,
    sector_scatter (std over sectors), and `cache` for the renderer.
    """
    n = len(frames)
    out = {k: np.full(n, np.nan) for k in
           ("v_raw", "v_clean", "frac_contaminated", "sector_scatter")}
    out["n_sect_rejected"] = np.zeros(n, int)
    cache = {}
    for k in range(n):
        if not tr["valid"][k]:
            continue
        cy, cx = tr["center"][k]
        rr, phi, ann, r0, r1 = annulus_geometry(
            frames[k].shape, cy, cx, tr["r_spot_px"][k], mode=mode, **geom)
        vr = radial_velocity(VX[k], VY[k], cy, cx, rr)
        bad = contamination_mask(frames[k], tr["masks"][k])
        ok = ann & np.isfinite(vr)
        if ok.sum() < 30:
            continue
        out["v_raw"][k] = np.nanmean(vr[ok])
        clean = ok & ~bad
        if clean.sum() > 30:
            out["v_clean"][k] = np.nanmean(vr[clean])
        out["frac_contaminated"][k] = (ann & bad).sum() / ann.sum()
        mid, vals, rej = sector_profile(vr, ann, phi, bad)
        out["n_sect_rejected"][k] = int(rej.sum())
        out["sector_scatter"][k] = np.nanstd(vals[~rej]) if (~rej).any() else np.nan
        cache[k] = (rr, phi, ann, r0, r1, vr, bad, mid, vals, rej)
    out["cache"] = cache
    return out


def render_audit_movie(frames, tr, VX, VY, t_h, t_iso, aud, out_mp4,
                       event_id="", fps=6, stride=1, half=110):
    """Write the verification movie. Requires ffmpeg."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle

    cache = aud["cache"]
    sample = frames[len(frames) // 2]
    ic_kw = dict(cmap="afmhot", origin="lower",
                 vmin=np.percentile(sample, 0.5),
                 vmax=np.percentile(sample, 99.9))

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1],
                          height_ratios=[1.1, 1, 0.55], hspace=0.38,
                          wspace=0.22)
    axIm = fig.add_subplot(gs[0:2, 0])
    axPol = fig.add_subplot(gs[0, 1], projection="polar")
    axRad = fig.add_subplot(gs[1, 1])
    axCur = fig.add_subplot(gs[2, :])
    writer = FFMpegWriter(fps=fps, bitrate=4500)

    with writer.saving(fig, str(out_mp4), dpi=105):
        for k in range(0, len(frames), stride):
            for ax in (axIm, axPol, axRad, axCur):
                ax.clear()
            axCur.plot(t_h, aud["v_raw"], "-", color="0.7", lw=1.1,
                       label="annulus mean, unmasked")
            axCur.plot(t_h, aud["v_clean"], "-", color="tab:blue", lw=1.3,
                       label="contamination masked (quoted)")
            axCur.axhline(0, color="k", lw=0.5)
            axCur.axvline(t_h[k], color="crimson", lw=1.2)
            axCur.set(xlabel="hours since start", ylabel="v$_r$ [km/s]")
            axCur.legend(fontsize=7.5, loc="upper left")
            axCur.grid(alpha=0.3)

            if k not in cache:
                axIm.text(.5, .5, "tracking lost at this epoch", ha="center",
                          transform=axIm.transAxes)
                for ax in (axIm, axPol, axRad):
                    ax.set_axis_off()
                fig.suptitle(f"{event_id} moat audit — t = {t_h[k]:.1f} h",
                             y=0.98, fontsize=11)
                writer.grab_frame()
                continue

            rr, phi, ann, r0, r1, vr, bad, mid, vals, rej = cache[k]
            cy, cx = tr["center"][k]
            axIm.imshow(frames[k], **ic_kw)
            axIm.contour(tr["masks"][k], levels=[.5], colors="lime", linewidths=1.0)
            if (bad & ann).any():
                axIm.contour(bad & ann, levels=[.5], colors="red", linewidths=0.7)
            s = 7
            Y, X = np.mgrid[0:vr.shape[0]:s, 0:vr.shape[1]:s]
            axIm.quiver(X, Y, VX[k][::s, ::s], VY[k][::s, ::s], color="c",
                        scale=10, width=0.0022)
            for r in (r0, r1):
                axIm.add_patch(Circle((cx, cy), r, fill=False, color="crimson",
                                      ls="--", lw=1.1))
            axIm.plot(cx, cy, "+", color="crimson", ms=11)
            axIm.set_xlim(cx - half, cx + half)
            axIm.set_ylim(cy - half, cy + half)
            axIm.set_axis_off()
            axIm.set_title("green = tracked spot | red = excluded (other "
                           "spots) | dashed = moat annulus", fontsize=9)

            width = 2 * np.pi / max(len(mid), 1)
            axPol.bar(mid, np.nan_to_num(vals), width=width * 0.9,
                      color=["0.75" if r else "tab:blue" for r in rej],
                      alpha=0.85)
            m = aud["v_clean"][k]
            th = np.linspace(-np.pi, np.pi, 100)
            axPol.plot(th, np.full_like(th, m), "k--", lw=1.0)
            axPol.set_ylim(min(-0.15, np.nanmin(vals) - .05) if np.isfinite(vals).any() else -0.15,
                           max(0.55, np.nanmax(vals) + .05) if np.isfinite(vals).any() else 0.55)
            axPol.set_title(f"v$_r$ per sector (mean {m:.2f} km/s, "
                            f"scatter {aud['sector_scatter'][k]:.2f})\n"
                            "grey = rejected, dashed = mean", fontsize=8.5)
            axPol.tick_params(labelsize=6)

            edges = np.arange(0, half, 3.0)
            rmid, rprof = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                sel = (rr >= a) & (rr < b) & ~bad & np.isfinite(vr)
                if sel.sum() > 20:
                    rmid.append(0.5 * (a + b) * PX_MM)
                    rprof.append(np.nanmean(vr[sel]))
            axRad.axhline(0, color="k", lw=0.5)
            axRad.plot(rmid, rprof, "o-", ms=3, color="tab:blue")
            axRad.axvspan(r0 * PX_MM, r1 * PX_MM, alpha=0.15, color="crimson")
            axRad.set(xlabel="r from spot centre [Mm]", ylabel="v$_r$ [km/s]",
                      ylim=(-0.3, 0.7))
            axRad.set_title("radial profile; shaded = annulus", fontsize=8.5)
            axRad.grid(alpha=0.3)

            fig.suptitle(f"{event_id} moat audit — t = {t_h[k]:.1f} h  "
                         f"({t_iso[k][:16]})   annulus contaminated: "
                         f"{aud['frac_contaminated'][k]*100:.0f}%",
                         y=0.98, fontsize=11)
            writer.grab_frame()
    plt.close(fig)
