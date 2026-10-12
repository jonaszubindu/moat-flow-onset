#!/usr/bin/env python
"""MMF onset from tracked magnetic features, compared with penumbra onset.

For each event: detect flux features in the 45 s magnetograms (every
--stride-th frame), track them, keep the ones that are born near the edge
of the tracked spot and move outward, and find when outward
OPPOSITE-polarity MMFs start (hourly birth rate, the same onset rule as
the other curves; moatflow.analysis.mmf). Run onset_analysis.py first --
the spot, its epochs and the penumbra onset come from there.

Usage:
  python scripts/mmf_onset.py AR11490 [AR11184 ...] [--stride 4]
  python scripts/mmf_onset.py --all                      # + sample table
  python scripts/mmf_onset.py AR11490 --movie            # + tracking movie
  python scripts/mmf_onset.py --all --v2 --movie         # version 2 (outputs *_v2)

Writes data/<event>/mmf_tracks.npz, quicklook/mmf_onset.json and
quicklook/mmf_onset.png (time-distance diagram of the tracks and the rate
curves with the penumbral area and onsets); with several events also
sample_results/mmf_onsets.csv, mmf_summary.txt and mmf_lags.png. With
--movie: quicklook/mmf_tracking.mp4 -- magnetogram and continuum around
the spot with every detected feature (green dots), outward MMFs circled
with their last hour's trail (red opposite, blue same polarity), the
excluded spot (yellow), edge / birth zone / limit circles, and the rate
curve with a time cursor.

--v2 (version 2, outputs *_v2, version 1 untouched): every track seen in
>= 3 frames and born within 10 Mm of the edge counts, classified only as
moving outward or inward; the MMF signal is outward minus inward per
hour, so detection noise and random motions cancel. Inward tracks get a
grey ring in the movie.
"""

import argparse
import csv
import io
import json
import sys
from contextlib import redirect_stdout

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from moatflow.analysis import mmf
from moatflow.analysis.eventdata import load_tracked_event
from moatflow.analysis.onsets import BASELINE_H, analyse, smooth
from moatflow.analysis.penumbra import segment_frame
from moatflow.analysis.spottrack import PX_MM
from moatflow.batch import add_event_args, resolve_events, run_batch
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes

M_CUBE = "cube_hmi.M_45s_magnetogram.h5"


def spot_polarity(d, mg, tm):
    """Sign of the median B_los over the umbra at the seed epoch."""
    tr, k = d["tr"], d["tr"]["seed_idx"]
    u, _ = segment_frame(d["frames"][k])
    umb = u & tr["masks"][k]
    j = int(np.argmin(np.abs(tm - d["t_mid"][k])))
    B = np.asarray(mg[j], dtype=float)
    b = float(np.nanmedian(B[umb])) if umb.any() else np.nan
    return (1 if b > 0 else -1), b


def run_event(cfg, eid, stride, movie=None, v2=False):
    """movie: None, or (frame step, fps) to also render mmf_tracking.mp4.
    v2: accept every track near the spot and use the outward-minus-inward
    excess (mmf.classify_v2); outputs carry the suffix _v2."""
    sfx = "_v2" if v2 else ""
    out = event_dir(cfg, eid)
    if not (out / M_CUBE).exists():
        raise SystemExit(f"{eid}: no {M_CUBE}")
    if not (out / "onset_series.npz").exists():
        raise SystemExit(f"{eid}: run onset_analysis.py first")
    with redirect_stdout(io.StringIO()):
        d = load_tracked_event(cfg, eid, with_mmf=False)
    tr, t_mid, t_h = d["tr"], d["t_mid"], d["t_h"]
    mg, t_mg, _ = load_cube(out / M_CUBE)
    tm = np.array([(t - d["t0"]).total_seconds() for t in parse_datetimes(t_mg)])
    pol, b_umb = spot_polarity(d, mg, tm)

    ok = tr["valid"]
    te = t_mid[ok]
    cy_e, cx_e, r_e = tr["center"][ok, 0], tr["center"][ok, 1], tr["r_spot_px"][ok]
    ke = np.flatnonzero(ok)
    sel = [j for j in range(0, len(tm), stride)
           if te[0] - 1800 <= tm[j] <= te[-1] + 1800
           and np.min(np.abs(te - tm[j])) <= 5400]
    excl_it = max(1, int(round(mmf.EXCL_MM / PX_MM)))
    tk = mmf.Tracker()
    finfo = []                     # light per-frame record, for the movie
    for j in sel:
        t = tm[j]
        cy, cx = np.interp(t, te, cy_e), np.interp(t, te, cx_e)
        rsp = np.interp(t, te, r_e)
        i = int(np.argmin(np.abs(te - t)))
        m0 = tr["masks"][ke[i]]
        mask = ndimage.shift(m0.astype(float), (cy - cy_e[i], cx - cx_e[i]),
                             order=0) > 0.5
        excl = ndimage.binary_dilation(mask, iterations=excl_it)
        B = np.asarray(mg[j], dtype=float)
        det = mmf.detect(B, excl, cy, cx, rsp, mmf.frame_noise(B))
        tk.add(t, det, cy, cx, rsp)
        finfo.append((t, j, cy, cx, rsp, ke[i], cy - cy_e[i], cx - cx_e[i]))
    tracks = tk.result()
    rows = mmf.classify_v2(tracks, pol) if v2 else mmf.classify(tracks, pol)

    t_sel = tm[sel] / 3600
    dt_h = np.median(np.diff(t_sel)) if len(t_sel) > 1 else 0.05
    covered = np.array([np.sum(np.abs(t_sel - t) < 0.5) >= 0.6 / dt_h for t in t_h])
    covered &= ok
    counts = {}
    if v2:
        # excess of outward over inward movers: noise cancels, MMFs remain
        for rp in ("opposite", "same"):
            for dr in ("out", "in"):
                counts[f"{rp}_{dr}"] = mmf.hourly_counts(rows, t_h, rp, dr, covered)
        rate_o = counts["opposite_out"] - counts["opposite_in"]
        rate_s = counts["same_out"] - counts["same_in"]
    else:
        rate_o = mmf.hourly_rate(rows, t_h, "opposite", covered)
        rate_s = mmf.hourly_rate(rows, t_h, "same", covered)
    on_o, on_s = mmf.mmf_onset(t_h, rate_o), mmf.mmf_onset(t_h, rate_s)
    whole = analyse(np.load(out / "onset_series.npz"))

    n_out_o = sum(r["outward"] and r["rel_pol"] == "opposite" for r in rows)
    n_out_s = sum(r["outward"] and r["rel_pol"] == "same" for r in rows)
    flags = []
    if n_out_o < 10:
        flags.append(f"few outward opposite-polarity MMFs ({n_out_o})")
    if np.isfinite(on_o["t"]) and on_o["t"] < on_o["t_first_valid"] + BASELINE_H:
        flags.append("MMF onset inside the baseline window (may already be on)")
    if not np.isfinite(on_o["t"]):
        flags.append("no MMF onset")
    res = {"event": eid, "spot_polarity": pol, "b_umbra_G": b_umb,
           "n_frames": len(sel), "cadence_min": float(dt_h * 60),
           "n_tracks": len(rows), "n_outward_opposite": int(n_out_o),
           "n_outward_same": int(n_out_s),
           "t_pen_h": whole["t_pen_h"], "t_pen_first_h": whole["t_pen_first_departure_h"],
           "t_moat_h": whole["t_moat_h"], "t_mmf_flct_h": whole["t_mmf_h"],
           "t_mmf_track_h": on_o["t"], "t_mmf_track_first_h": on_o["t_first"],
           "t_mmf_same_h": on_s["t"],
           "lag_mmf_track_h": on_o["t"] - whole["t_pen_h"],
           "lag_mmf_track_first_h": on_o["t_first"] - whole["t_pen_first_departure_h"],
           "lag_mmf_first_vs_pen_h": on_o["t_first"] - whole["t_pen_h"],
           "mmf_rate_baseline_per_h": on_o["zero"], "mmf_rate_mature_per_h": on_o["mature"],
           "mmf_holds": on_o["holds"], "flags": flags, "version": 2 if v2 else 1,
           "n_inward_opposite": int(sum(r.get("dir") == "in" and r["rel_pol"] == "opposite"
                                        for r in rows)) if v2 else None,
           "n_inward_same": int(sum(r.get("dir") == "in" and r["rel_pol"] == "same"
                                    for r in rows)) if v2 else None,
           "median_v_kms": float(np.nanmedian([r["v_kms"] for r in rows
                                               if r["outward"] and r["rel_pol"] == "opposite"]))
           if n_out_o else np.nan}

    keys = ["t_birth", "life_min", "d_birth_mm", "v_kms", "disp_mm", "flux_Mx"]
    np.savez(out / f"mmf_tracks{sfx}.npz", t_h=t_h, rate_opposite=rate_o, rate_same=rate_s,
             **{f"count_{k}": v for k, v in counts.items()},
             direction=np.array([str(r.get("dir")) for r in rows]),
             covered=covered, rel_pol=np.array([r["rel_pol"] for r in rows]),
             outward=np.array([r["outward"] for r in rows]),
             # all positions: track id, t [s], x, y [px], distance from edge [Mm]
             pos_track=np.concatenate([np.full(len(tr["t"]), i) for i, tr in enumerate(tracks)]),
             pos_t=np.concatenate([tr["t"] for tr in tracks]),
             pos_x=np.concatenate([tr["x"] for tr in tracks]),
             pos_y=np.concatenate([tr["y"] for tr in tracks]),
             pos_d=np.concatenate([tr["d_edge_mm"] for tr in tracks]),
             **{k: np.array([r[k] for r in rows], dtype=float) for k in keys})
    (out / "quicklook").mkdir(exist_ok=True)
    with open(out / "quicklook" / f"mmf_onset{sfx}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    plot_event(out / "quicklook" / f"mmf_onset{sfx}.png", eid, tracks, rows, t_h,
               rate_o, rate_s, on_o, on_s, whole, d["pf_lit"],
               np.load(out / "onset_series.npz"), res, counts)
    if movie:
        mp4 = out / "quicklook" / f"mmf_tracking{sfx}.mp4"
        render_movie(mp4, eid, d, mg, finfo, tracks, rows, t_h, on_o, on_s,
                     np.load(out / "onset_series.npz"), whole, movie[0], movie[1])
        print(f"   movie -> {mp4}")
    print(f"{eid}{' [v2]' if v2 else ''}: spot {'+' if pol > 0 else '-'} ({b_umb:+.0f} G), "
          f"{len(sel)} frames at {dt_h * 60:.1f} min, {len(rows)} tracks, outward: "
          f"{n_out_o} opposite / {n_out_s} same polarity"
          + (f"; inward: {res['n_inward_opposite']} opposite / {res['n_inward_same']} same"
             if v2 else ""))
    print(f"   penumbra onset {whole['t_pen_h']:.1f} h | MMF (tracked, opposite) "
          f"onset {on_o['t']:.1f} h, first {on_o['t_first']:.1f} h | FLCT MMF "
          f"{whole['t_mmf_h']:.1f} h  ->  lag {res['lag_mmf_track_h']:+.1f} h")
    for fl in flags:
        print(f"   flag: {fl}")
    return res


def render_movie(mp4, eid, d, mg, finfo, tracks, rows, t_h, on_o, on_s, npz,
                 whole, step, fps):
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle
    from moatflow.batch import CUBE_IC
    from moatflow.cubes import load_cube as _lc
    from moatflow.viz import parse_times
    ic, ic_iso, _ = _lc(d["out"] / CUBE_IC)
    t_ic = parse_times(ic_iso)
    tr_masks = d["tr"]["masks"]
    # frame time -> [(track index, x, y)]
    at = {}
    for k, tr in enumerate(tracks):
        for t, x, y in zip(tr["t"], tr["x"], tr["y"]):
            at.setdefault(float(t), []).append((k, x, y))
    cls = ["opp" if (r["outward"] and r["rel_pol"] == "opposite") else
           "same" if r["outward"] else "in" if r.get("dir") == "in" else None
           for r in rows]
    half = (np.nanmax([f[4] for f in finfo]) * PX_MM + mmf.R_MAX_MM + 2) / PX_MM
    sample_ic = np.asarray(ic[len(t_ic) // 2], dtype=float)
    ic_kw = dict(cmap="gray", vmin=np.nanpercentile(sample_ic, 1),
                 vmax=np.nanpercentile(sample_ic, 99.5), origin="lower")
    fig = plt.figure(figsize=(13, 8.6))
    g = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], hspace=0.22, wspace=0.08)
    aB, aI, aC = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1]), fig.add_subplot(g[1, :])
    v2 = any("dir" in r for r in rows)
    aC.plot(t_h, on_o["smooth"], color="crimson", lw=1.5,
            label="opposite polarity: outward - inward / h" if v2 else
            "outward opposite-polarity MMFs / h")
    aC.plot(t_h, on_s["smooth"], color="royalblue", lw=1.0,
            label="same polarity: outward - inward / h" if v2 else
            "outward same-polarity MMFs / h")
    if v2:
        aC.axhline(0, color="k", lw=0.5)
    aC.set_ylabel("excess births / h" if v2 else "MMF births / h")
    a3 = aC.twinx()
    a3.plot(npz["t_h"], smooth(np.where(npz["valid"].astype(bool), npz["a_pen"], np.nan)),
            color="tab:orange", lw=1.2)
    a3.set_ylabel("penumbra [Mm$^2$]", color="tab:orange")
    for tt, col, ls in ((whole["t_pen_h"], "tab:orange", "--"), (on_o["t"], "crimson", "--"),
                        (on_o["t_first"], "crimson", ":")):
        if np.isfinite(tt):
            aC.axvline(tt, color=col, ls=ls, lw=1.0)
    cursor = aC.axvline(np.nan, color="k", lw=1.2)
    aC.legend(fontsize=7, loc="upper left")
    aC.set_xlabel("hours since start")
    aC.grid(alpha=0.3)
    births = np.array([r["t_birth"] for r, c in zip(rows, cls) if c == "opp"])
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    with writer.saving(fig, str(mp4), dpi=100):
        for t, j, cy, cx, rsp, ke, sy, sx in finfo[::step]:
            B = np.asarray(mg[j], dtype=float)
            I = np.asarray(ic[int(np.argmin(np.abs(t_ic - t)))], dtype=float)
            mask = ndimage.shift(tr_masks[ke].astype(float), (sy, sx), order=0) > 0.5
            excl = ndimage.binary_dilation(mask, iterations=max(1, int(round(mmf.EXCL_MM / PX_MM))))
            here = at.get(float(t), [])
            for ax, img, kw in ((aB, B, dict(cmap="gray", vmin=-150, vmax=150, origin="lower")),
                                (aI, I, ic_kw)):
                ax.clear()
                ax.imshow(img, interpolation="nearest", **kw)
                ax.contour(excl, levels=[0.5], colors="gold", linewidths=0.8)
                for r_mm, ls, col in ((0, "--", "w"), (mmf.BIRTH_MM, ":", "orange"),
                                      (mmf.R_MAX_MM, ":", "0.6")):
                    ax.add_patch(Circle((cx, cy), rsp + r_mm / PX_MM, fill=False,
                                        ls=ls, color=col, lw=0.9))
                if here:
                    xs, ys = zip(*[(x, y) for _, x, y in here])
                    ax.plot(xs, ys, ".", color="lime", ms=2.5, alpha=0.7)
                for k, x, y in here:
                    c = cls[k]
                    if c is None:
                        continue
                    col = {"opp": "red", "same": "deepskyblue", "in": "0.75"}[c]
                    tt = tracks[k]["t"]
                    m = (tt <= t) & (tt >= t - 3600)
                    ax.plot(tracks[k]["x"][m], tracks[k]["y"][m], "-", color=col, lw=1.0)
                    ax.plot(x, y, "o", mfc="none", mec=col, ms=8 if c != "in" else 6,
                            mew=1.3 if c != "in" else 0.8)
                ax.set_xlim(cx - half, cx + half)
                ax.set_ylim(cy - half, cy + half)
                ax.set_xticks([])
                ax.set_yticks([])
            n_on = sum(1 for k, _, _ in here if cls[k] == "opp")
            aB.set_title(f"magnetogram +-150 G: outward now {n_on} opposite (red), "
                         f"{sum(1 for k, _, _ in here if cls[k] == 'same')} same (blue)"
                         + (f", inward {sum(1 for k, _, _ in here if cls[k] == 'in')} (grey)"
                            if v2 else ""), fontsize=9)
            aI.set_title("continuum, same overlays (yellow: excluded spot)", fontsize=9)
            cursor.set_xdata([t / 3600, t / 3600])
            fig.suptitle(f"{eid}  t = {t / 3600:.1f} h   outward opposite-polarity MMFs born so far: "
                         f"{int(np.sum(births <= t))}   (circles: spot edge, +{mmf.BIRTH_MM:.0f} Mm "
                         f"birth zone, +{mmf.R_MAX_MM:.0f} Mm limit)", fontsize=10, y=0.995)
            writer.grab_frame()
    plt.close(fig)


def plot_event(png, eid, tracks, rows, t_h, rate_o, rate_s, on_o, on_s, whole,
               pf, npz, res, counts=None):
    v2 = bool(counts)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1]})
    for tr, r in zip(tracks, rows):
        if not r["outward"]:
            if (r.get("dir") == "in" if v2 else
                    (r["rel_pol"] == "opposite" and r["life_min"] >= mmf.MIN_LIFE_MIN)):
                a1.plot(tr["t"] / 3600, tr["d_edge_mm"], color="0.8", lw=0.4)
            continue
        col = "crimson" if r["rel_pol"] == "opposite" else "royalblue"
        a1.plot(tr["t"] / 3600, tr["d_edge_mm"], color=col, lw=0.9, alpha=0.8)
    a1.axhline(0, color="k", lw=0.6)
    a1.set(ylim=(-mmf.EXCL_MM - 1, mmf.R_MAX_MM), ylabel="distance from spot edge [Mm]",
           title=(f"{eid} [v2]: every track born within {mmf.BIRTH_MM_V2:.0f} Mm, >= "
                  f"{mmf.MIN_SAMPLES_V2} frames -- outward opposite (red), outward same "
                  f"(blue), inward (grey)" if v2 else
                  f"{eid}: tracked MMFs -- outward opposite polarity (red), outward "
                  f"same polarity (blue), other opposite-polarity tracks (grey)"))
    a2.plot(t_h, rate_o, ".", color="crimson", ms=4, alpha=0.5)
    if v2:
        a2.plot(t_h, smooth(counts["opposite_out"]), "--", color="crimson", lw=0.9,
                label=f"opposite, outward ({res['n_outward_opposite']})")
        a2.plot(t_h, smooth(counts["opposite_in"]), "--", color="0.5", lw=0.9,
                label=f"opposite, inward ({res['n_inward_opposite']})")
        a2.plot(t_h, on_o["smooth"], "-", color="crimson", lw=1.8,
                label="opposite: outward - inward (MMF signal)")
        a2.plot(t_h, on_s["smooth"], "-", color="royalblue", lw=1.2,
                label="same: outward - inward")
        a2.axhline(0, color="k", lw=0.6)
    else:
        a2.plot(t_h, on_o["smooth"], "-", color="crimson", lw=1.8,
                label=f"outward opposite-polarity MMFs ({res['n_outward_opposite']})")
        a2.plot(t_h, on_s["smooth"], "-", color="royalblue", lw=1.2,
                label=f"outward same-polarity MMFs ({res['n_outward_same']})")
    if np.isfinite(on_o["thr"]):
        a2.axhline(on_o["thr"], color="crimson", lw=0.7, ls=":")
    a2.set_ylabel("births per hour" + (" (excess)" if v2 else ""))
    ax3 = a2.twinx()
    a = np.where(npz["valid"].astype(bool), npz["a_pen"], np.nan)
    ax3.plot(npz["t_h"], smooth(a), color="tab:orange", lw=1.4)
    ax3.set_ylabel("penumbral area [Mm$^2$]", color="tab:orange")
    for ax in (a1, a2):
        if np.isfinite(pf[0]):
            ax.axvspan(pf[0], pf[1] if np.isfinite(pf[1]) else pf[0] + 0.3,
                       color="green", alpha=0.10)
        for tt, col, ls in ((whole["t_pen_h"], "tab:orange", "--"),
                            (on_o["t"], "crimson", "--"),
                            (on_o["t_first"], "crimson", ":"),
                            (whole["t_mmf_h"], "tab:purple", "-.")):
            if np.isfinite(tt):
                ax.axvline(tt, color=col, ls=ls, lw=1.1)
    a2.plot([], [], "--", color="tab:orange", label=f"penumbra onset {whole['t_pen_h']:.1f} h")
    a2.plot([], [], "--", color="crimson", label=f"MMF onset (tracked) {on_o['t']:.1f} h")
    a2.plot([], [], ":", color="crimson", label=f"MMF first appearance {on_o['t_first']:.1f} h")
    a2.plot([], [], "-.", color="tab:purple", label=f"FLCT MMF onset {whole['t_mmf_h']:.1f} h")
    a2.legend(fontsize=7, loc="upper left", ncol=2)
    a2.set_xlabel("hours since start")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def summary(cfg, ids, sfx=""):
    rd = cfg["data_root"] / "sample_results"
    rd.mkdir(exist_ok=True)
    used = {}
    f = rd / "onsets.csv"
    if f.exists():
        used = {r["event"]: r["used"] == "True" for r in csv.DictReader(open(f))}
    res = []
    for eid in ids:
        j = event_dir(cfg, eid) / "quicklook" / f"mmf_onset{sfx}.json"
        if j.exists():
            r = json.load(open(j))
            r["used"] = used.get(eid, False)
            res.append(r)
    if not res:
        return
    cols = ["event", "used", "spot_polarity", "n_outward_opposite", "n_outward_same",
            "median_v_kms", "t_pen_h", "t_mmf_track_h", "lag_mmf_track_h",
            "t_pen_first_h", "t_mmf_track_first_h", "lag_mmf_track_first_h",
            "lag_mmf_first_vs_pen_h",
            "t_mmf_flct_h", "t_moat_h", "mmf_holds", "flags"]
    with open(rd / f"mmf_onsets{sfx}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in res:
            w.writerow([("; ".join(r[c]) if c == "flags" else r.get(c)) for c in cols])
    buf = io.StringIO()

    def fmt(x, sign=False):
        return "-" if x is None or not np.isfinite(x) else (f"{x:+.1f}" if sign else f"{x:.1f}")
    print("MMF onset from tracked outward opposite-polarity features vs penumbra onset"
          + ("  [v2: every track, outward-minus-inward excess]" if sfx else ""), file=buf)
    print("(lag = MMF - penumbra; 1st = first appearance of each; MMF1st-pen = MMF first\n"
          " appearance minus the 40 % penumbra onset)\n", file=buf)
    print(f"{'event':10s} used  nMMF  v[km/s]   pen   MMF    lag | pen1st MMF1st  lag | MMF1st-pen | FLCT-MMF  moat",
          file=buf)
    for r in res:
        print(f"{r['event']:10s} {'yes' if r['used'] else 'no ':4s} {r['n_outward_opposite']:5d}  "
              f"{fmt(r['median_v_kms']):>6s}  {fmt(r['t_pen_h']):>5s} {fmt(r['t_mmf_track_h']):>5s} "
              f"{fmt(r['lag_mmf_track_h'], True):>6s} | {fmt(r['t_pen_first_h']):>5s} "
              f"{fmt(r['t_mmf_track_first_h']):>6s} {fmt(r['lag_mmf_track_first_h'], True):>5s} | "
              f"{fmt(r.get('lag_mmf_first_vs_pen_h', np.nan), True):>10s} | "
              f"{fmt(r['t_mmf_flct_h']):>7s} {fmt(r['t_moat_h']):>5s}", file=buf)
        for fl in r["flags"]:
            print(f"{'':12s}flag: {fl}", file=buf)
    lags = [r["lag_mmf_track_h"] for r in res if r["used"] and not r["flags"]
            and np.isfinite(r["lag_mmf_track_h"])]
    for key, lab in (("lag_mmf_track_h", "MMF onset - penumbra onset (40 % rules)"),
                     ("lag_mmf_first_vs_pen_h", "MMF first appearance - penumbra onset")):
        v = [r.get(key, np.nan) for r in res if r["used"] and not r["flags"]]
        v = [x for x in v if np.isfinite(x)]
        if v:
            print(f"\n{lab}, used events without flags (n = {len(v)}): median {np.median(v):+.1f} h, "
                  f"{sum(x > 2 for x in v)} after / {sum(abs(x) <= 2 for x in v)} within 2 h / "
                  f"{sum(x < -2 for x in v)} before", file=buf)
    txt = buf.getvalue()
    print("\n" + txt)
    (rd / f"mmf_summary{sfx}.txt").write_text(txt)

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(res) + 1.5))
    for i, r in enumerate(res):
        c = "k" if r["used"] and not r["flags"] else "0.6"
        for key, mk in (("lag_mmf_track_h", "o"), ("lag_mmf_first_vs_pen_h", "^")):
            if np.isfinite(r[key]):
                ax.plot(r[key], i, mk, color=c, mfc=c if mk == "o" else "none", ms=7)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(res)), [r["event"] for r in res])
    ax.set_xlabel("MMF - penumbra onset [h]  (o MMF 40 % onset, ^ MMF first appearance; grey: excluded/flagged)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rd / f"mmf_lags{sfx}.png", dpi=130)
    plt.close(fig)
    print(f"-> {rd / f'mmf_onsets{sfx}.csv'}, mmf_summary{sfx}.txt, mmf_lags{sfx}.png")


def main():
    ap = argparse.ArgumentParser()
    add_event_args(ap)
    ap.add_argument("--stride", type=int, default=4,
                    help="use every Nth 45 s magnetogram (4 -> 3 min)")
    ap.add_argument("--movie", action="store_true",
                    help="also render quicklook/mmf_tracking.mp4")
    ap.add_argument("--movie-step", type=int, default=1,
                    help="movie uses every Nth tracked frame (1 -> 3 min steps)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--v2", action="store_true",
                    help="version 2: every track near the spot, outward-minus-inward "
                         "excess; outputs *_v2 (version 1 untouched)")
    args = ap.parse_args()
    cfg = load_config()
    ids = resolve_events(args, cfg, ap)
    movie = (args.movie_step, args.fps) if args.movie else None
    sfx = "_v2" if args.v2 else ""
    code = run_batch(lambda e: run_event(cfg, e, args.stride, movie, args.v2), ids,
                     out_name=f"mmf_onset{sfx}.png", cfg=cfg,
                     skip_existing=args.skip_existing)
    if len(ids) > 1:
        summary(cfg, ids, sfx)
    sys.exit(code)


if __name__ == "__main__":
    main()
