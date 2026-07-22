"""Quicklook and vetting: movies, tracking-drift check, cadence report.

Vetting answers three questions before any tracking is trusted:
  1. Does the spot stay fixed in the frame (im_patch / SHARP tracking OK)?
     -> drift_series: cumulative frame-to-frame registration; residual
        drift shows up directly in the LCT velocities as a bias flow.
  2. Are there missing or bad frames? -> cadence_report.
  3. Does the scene look sane (intensity range, no garbage)? ->
     summary_figure + movie.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Frame registration / drift
# ---------------------------------------------------------------------------
def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dy, dx) shift of b relative to a via FFT cross-correlation,
    refined to subpixel with a parabolic fit around the peak."""
    a = np.nan_to_num(a - np.nanmean(a))
    b = np.nan_to_num(b - np.nanmean(b))
    cc = np.fft.irfft2(np.fft.rfft2(a).conj() * np.fft.rfft2(b), s=a.shape)
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)

    def _sub(c, i, n):
        m, p = c.take((i - 1) % n), c.take((i + 1) % n)
        denom = 2 * c.take(i) - m - p
        return 0.0 if denom == 0 else 0.5 * (m - p) / denom

    dy = iy + _sub(cc[:, ix], iy, cc.shape[0])
    dx = ix + _sub(cc[iy, :], ix, cc.shape[1])
    ny, nx = a.shape
    if dy > ny / 2:
        dy -= ny
    if dx > nx / 2:
        dx -= nx
    return float(dy), float(dx)


def drift_series(cube, stride: int = 1) -> np.ndarray:
    """Cumulative (n, 2) [dy, dx] drift from consecutive-frame registration.

    Consecutive frames correlate well even for evolving granulation; the
    cumulative sum exposes slow tracking drift that direct registration
    against frame 0 would lose once the scene has evolved.
    """
    idx = list(range(0, len(cube), stride))
    steps = [phase_shift(np.asarray(cube[i]), np.asarray(cube[j]))
             for i, j in zip(idx[:-1], idx[1:])]
    return np.concatenate([[[0.0, 0.0]], np.cumsum(steps, axis=0)])


def plot_drift(drift: np.ndarray, times_h: np.ndarray, out_png: Path,
               title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times_h, drift[:, 1], label="dx [px]")
    ax.plot(times_h, drift[:, 0], label="dy [px]")
    ax.set(xlabel="time [h]", ylabel="cumulative drift [px]", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cadence / gap report
# ---------------------------------------------------------------------------
def parse_datetimes(t_iso: list[str]) -> list[datetime]:
    """ISO or JSOC-style ('2012.05.27_02:00:06.657_TAI') -> datetimes.
    Only the date part has its dots replaced — the time part may carry
    fractional seconds."""
    out = []
    for t in t_iso:
        t = str(t).replace("_TAI", "")
        if "_" in t:
            date, time = t.split("_", 1)
            t = date.replace(".", "-") + "T" + time
        out.append(datetime.fromisoformat(t))
    return out


def parse_times(t_iso: list[str]) -> np.ndarray:
    """ISO or JSOC-style strings -> seconds since first frame."""
    ts = parse_datetimes(t_iso)
    return np.array([(t - ts[0]).total_seconds() for t in ts])


def cadence_report(times_s: np.ndarray, verbose: bool = True) -> dict:
    dt = np.diff(times_s)
    cad = float(np.median(dt))
    gaps = np.flatnonzero(dt > 1.5 * cad)
    rep = {
        "n_frames": len(times_s),
        "cadence_s": cad,
        "duration_h": float(times_s[-1] / 3600),
        "n_gaps": len(gaps),
        "gaps": [(int(i), float(dt[i])) for i in gaps],
        "missing_frames": int(round(sum(dt[gaps] / cad - 1))),
    }
    if verbose:
        print(f"  {rep['n_frames']} frames, cadence {cad:.0f} s, "
              f"{rep['duration_h']:.1f} h, {rep['n_gaps']} gaps "
              f"(~{rep['missing_frames']} missing frames)")
        for i, g in rep["gaps"][:10]:
            print(f"    gap after frame {i}: {g:.0f} s")
    return rep


# ---------------------------------------------------------------------------
# Figures / movie
# ---------------------------------------------------------------------------
def _limits(cube, segment: str, nsample: int = 20):
    sample = np.concatenate([np.asarray(cube[i]).ravel()
                             for i in np.linspace(0, len(cube) - 1, nsample,
                                                  dtype=int)])
    sample = sample[np.isfinite(sample)]
    if "magnetogram" in segment:
        vmax = np.percentile(np.abs(sample), 99.5)
        return dict(cmap="gray", vmin=-vmax, vmax=vmax)
    return dict(cmap="afmhot", vmin=np.percentile(sample, 0.5),
                vmax=np.percentile(sample, 99.9))


def summary_figure(cube, times_h: np.ndarray, out_png: Path,
                   segment: str = "continuum", title: str = "") -> None:
    """First/middle/last frame + mean-signal time series."""
    kw = _limits(cube, segment)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, i in zip(axes[:3], [0, len(cube) // 2, len(cube) - 1]):
        ax.imshow(np.asarray(cube[i]), origin="lower", **kw)
        ax.set_title(f"frame {i} ({times_h[i]:.1f} h)")
        ax.set_axis_off()
    mean = [float(np.nanmean(np.abs(np.asarray(cube[i]))))
            for i in range(0, len(cube), max(1, len(cube) // 400))]
    axes[3].plot(np.linspace(times_h[0], times_h[-1], len(mean)), mean)
    axes[3].set(xlabel="time [h]", ylabel="mean |signal|",
                title="frame-mean (bad frames show as spikes)")
    axes[3].grid(alpha=0.3)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_movie(cube, t_iso: list[str], out_mp4: Path,
               segment: str = "continuum", fps: int = 20,
               max_frames: int = 600, title: str = "") -> None:
    """MP4 quicklook movie (frames subsampled to at most max_frames)."""
    from matplotlib.animation import FFMpegWriter

    stride = max(1, len(cube) // max_frames)
    idx = range(0, len(cube), stride)
    kw = _limits(cube, segment)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.asarray(cube[0]), origin="lower", **kw)
    ax.set_axis_off()
    txt = ax.set_title("")
    writer = FFMpegWriter(fps=fps, bitrate=3000)
    with writer.saving(fig, str(out_mp4), dpi=120):
        for i in idx:
            im.set_data(np.asarray(cube[i]))
            txt.set_text(f"{title}  {t_iso[i]}")
            writer.grab_frame()
    plt.close(fig)
    print(f"  wrote {out_mp4} ({len(list(idx))} frames, stride {stride})")
