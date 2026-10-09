#!/usr/bin/env python
"""Overview plots and movies of the ME results written by me_invert.py.

  python sst/me_quicklook.py me_AR13010 [--scan 40] [--movie] [--fps 6]

Writes into <dir>/quicklook/:
  overview_scanNNNN.png   9 panels: far-wing intensity, |B|, B_los, B_trans,
                          inclination, azimuth, v_los, Doppler width, chi2
  movie_me.mp4            per scan: intensity (granulation), B_los
                          (+-1500 G), B_los saturated at +-150 G (weak
                          flux: MMFs, network), v_los
v_los is shown relative to the quiet-Sun median of each scan (header VQS);
the absolute zero point (convective blueshift, wavelength calibration) is
not fixed here. Azimuth and axes are in the image frame of the cube (SST
images are not necessarily rotated to solar north).
"""

import argparse
import glob
import os
import pathlib
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def load(path):
    with fits.open(path) as h:
        d, hd = h[0].data.astype(float), h[0].header
    names = [hd[f"PAR{i:02d}"] for i in range(d.shape[0])]
    return {n: d[i] for i, n in enumerate(names)}, hd


def find_ffmpeg():
    """First ffmpeg that actually runs (an x86 build can be installed but
    unusable on Apple silicon)."""
    cands = [os.environ.get("FFMPEG"), shutil.which("ffmpeg"),
             *glob.glob(str(pathlib.Path.home() / "anaconda3/envs/*/bin/ffmpeg")),
             "/opt/homebrew/bin/ffmpeg"]
    for c in cands:
        if c and os.path.exists(c):
            try:
                subprocess.run([c, "-version"], capture_output=True, check=True)
                return c
            except (OSError, subprocess.CalledProcessError):
                continue
    return None


def pct(a, lo=1, hi=99):
    g = np.isfinite(a)
    return np.percentile(a[g], [lo, hi]) if g.any() else (0, 1)


def overview(m, hd, png):
    ny, nx = m["B"].shape
    s = hd.get("PIXSCALE", 1.0)
    ext = [0, nx * s, 0, ny * s]
    v = m["vlos"] - hd.get("VQS", 0.0)
    panels = [
        ("I_wing", "intensity (far wing) / I$_{QS}$", "gray", pct(m["I_wing"])),
        ("B", "|B| [G]", "magma", (0, pct(m["B"], 0, 99.5)[1])),
        ("Blos", "B$_{los}$ [G]", "RdBu_r", (-1500, 1500)),
        ("Btrans", "B$_{trans}$ [G]", "viridis", (0, pct(m["Btrans"], 0, 99.5)[1])),
        ("inc_deg", "inclination [deg]", "RdBu_r", (0, 180)),
        ("azi_deg", "azimuth [deg] (image frame)", "twilight", (0, 180)),
        (None, "v$_{los}$ - v$_{QS}$ [km/s] (+ = red)", "RdBu_r", (-3, 3)),
        ("vdop", "Doppler width [A]", "cividis", pct(m["vdop"])),
        ("chi2", "log$_{10}$ $\\chi^2$", "inferno", None),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(15, 13.5 * ny / nx + 1.5),
                            sharex=True, sharey=True)
    for ax, (key, lab, cmap, lim) in zip(axs.flat, panels):
        img = v if key is None else (np.log10(np.maximum(m["chi2"], 1e-3))
                                     if key == "chi2" else m[key])
        lim = pct(img) if lim is None else lim
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1],
                       extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=lab)
    for ax in axs[-1]:
        ax.set_xlabel("x [arcsec]")
    for ax in axs[:, 0]:
        ax.set_ylabel("y [arcsec]")
    fig.suptitle(f"{hd.get('CUBE', '')}  scan {hd.get('SCAN')}  "
                 f"{hd.get('DATE-OBS', '')[:19]}  mu = {hd.get('MU', np.nan):.2f}  "
                 f"(x, y) = ({hd.get('XCEN', np.nan):.0f}\", {hd.get('YCEN', np.nan):.0f}\")",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def movie(files, mp4, fps, ffmpeg):
    from matplotlib.animation import FFMpegWriter
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg
    m0, hd0 = load(files[0])
    ny, nx = m0["B"].shape
    s = hd0.get("PIXSCALE", 1.0)
    ext = [0, nx * s, 0, ny * s]
    ilim = pct(m0["I_wing"])
    spec = [("I_wing", "intensity (far wing): granulation", "gray", ilim),
            ("Blos", "B$_{los}$ [G]", "RdBu_r", (-1500, 1500)),
            ("Blos", "B$_{los}$ saturated +-150 G: MMFs, network", "RdBu_r", (-150, 150)),
            ("v", "v$_{los}$ - v$_{QS}$ [km/s]", "RdBu_r", (-3, 3))]
    fig, axs = plt.subplots(2, 2, figsize=(12, 12 * ny / nx + 1.0),
                            sharex=True, sharey=True)
    ims = []
    for ax, (key, lab, cmap, lim) in zip(axs.flat, spec):
        img = m0["vlos"] - hd0.get("VQS", 0.0) if key == "v" else m0[key]
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1],
                       extent=ext, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(lab, fontsize=10)
        ims.append((im, key))
    title = fig.suptitle("", y=0.995)
    fig.tight_layout()
    writer = FFMpegWriter(fps=fps, bitrate=6000)
    with writer.saving(fig, str(mp4), dpi=100):
        for f in files:
            m, hd = load(f)
            for im, key in ims:
                im.set_data(m["vlos"] - hd.get("VQS", 0.0) if key == "v" else m[key])
            title.set_text(f"scan {hd.get('SCAN')}  {hd.get('DATE-OBS', '')[:19]}  "
                           f"mu = {hd.get('MU', np.nan):.2f}")
            writer.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="output directory of me_invert.py")
    ap.add_argument("--scan", type=int, default=None,
                    help="scan for the overview (default: middle of the series)")
    ap.add_argument("--all-overviews", action="store_true",
                    help="an overview PNG for every scan")
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "me_scan*.fits")))
    if not files:
        raise SystemExit(f"no me_scan*.fits in {args.dir}")
    ql = pathlib.Path(args.dir) / "quicklook"
    ql.mkdir(exist_ok=True)
    scans = {int(pathlib.Path(f).stem[7:]): f for f in files}
    pick = (list(scans) if args.all_overviews else
            [args.scan if args.scan is not None else sorted(scans)[len(scans) // 2]])
    for k in pick:
        m, hd = load(scans[k])
        png = ql / f"overview_scan{k:04d}.png"
        overview(m, hd, png)
        print(f"-> {png}")
    if args.movie:
        ff = find_ffmpeg()
        if ff is None:
            raise SystemExit("no working ffmpeg found (set FFMPEG=/path/to/ffmpeg)")
        mp4 = ql / "movie_me.mp4"
        movie(files, mp4, args.fps, ff)
        print(f"-> {mp4}  ({len(files)} scans, ffmpeg {ff})")


if __name__ == "__main__":
    main()
