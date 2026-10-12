#!/usr/bin/env python
"""Is the moat hidden before the penumbra? Radial flow structure vs time.

Test 1 (granulation): azimuthal mean FLCT v_r against distance from the
spot edge, per hour. Version "inflow ring" (Rempel 2015; Deng et al.
2007) predicts, before the penumbra onset, an inflow at the edge (0-3 Mm)
with outflow farther out (5-12 Mm), the inflow vanishing and the outflow
reaching the edge as the penumbra grows. A moat hidden below the
granulation shows no such structure: the outflow appears at all
distances together.
Test 2 (magnetic elements): radial drift of long-lived tracked flux
features on the same axes (needs mmf_tracks.npz from mmf_onset.py; with
and without the outward MMFs). Features anchored deeper than granules
could drift outward before the granular moat appears.

Usage:
  python scripts/radial_test.py AR11490 [AR11184 ...]
  python scripts/radial_test.py --all          # + stack over the used events

Per event: radial_test.npz, quicklook/radial_test.json and
quicklook/radial_test.png. With several events:
sample_results/radial_test_stack.png and radial_test_summary.txt.
Windows (hours from the whole-spot penumbra onset): pre -14..-2,
onset -2..+4, post +6..+18.
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

from moatflow.analysis import radial as R
from moatflow.analysis.eventdata import load_tracked_event
from moatflow.analysis.onsets import analyse
from moatflow.batch import add_event_args, resolve_events, run_batch
from moatflow.config import event_dir, load_config

WCOL = {"pre": "tab:blue", "onset": "tab:green", "post": "tab:red"}


def run_event(cfg, eid):
    out = event_dir(cfg, eid)
    if not (out / "onset_series.npz").exists():
        raise SystemExit(f"{eid}: run onset_analysis.py first")
    with redirect_stdout(io.StringIO()):
        d = load_tracked_event(cfg, eid, with_mmf=False)
    t_h = d["t_h"]
    gmap = R.granular_map(d["frames"], d["tr"], d["VX"], d["VY"])
    on = analyse(np.load(out / "onset_series.npz"))
    t_pen = on["t_pen_h"]
    mfile = out / "mmf_tracks.npz"
    mag = mag_x = None
    if mfile.exists():
        mt = np.load(mfile)
        if "pos_track" in mt.files:
            mag, mag_n, mag_s = R.magnetic_drift_map(mt, t_h)
            mag_x, magx_n, magx_s = R.magnetic_drift_map(mt, t_h, exclude_outward_mmf=True)
        else:
            print(f"  {eid}: mmf_tracks.npz predates saved positions -- rerun "
                  "mmf_onset.py for test 2")
    res = {"event": eid, "t_pen_h": t_pen, "t_moat_h": on["t_moat_h"],
           "has_magnetic": mag is not None}
    np.savez(out / "radial_test.npz", t_h=t_h, d_mid=R.D_MID, granular=gmap,
             magnetic=mag if mag is not None else np.full_like(gmap, np.nan),
             magnetic_no_mmf=mag_x if mag_x is not None else np.full_like(gmap, np.nan),
             mag_sums=mag_s if mag is not None else np.zeros_like(gmap),
             mag_counts=mag_n if mag is not None else np.zeros_like(gmap),
             magx_sums=magx_s if mag is not None else np.zeros_like(gmap),
             magx_counts=magx_n if mag is not None else np.zeros_like(gmap),
             t_pen_h=t_pen, t_moat_h=on["t_moat_h"])
    if np.isfinite(t_pen):
        t_rel = t_h - t_pen
        wg = R.window_profiles(t_rel, gmap)
        for w, p in wg.items():
            res[f"gran_near_{w}"] = R.band_mean(p, R.NEAR_MM)
            res[f"gran_outer_{w}"] = R.band_mean(p, R.OUTER_MM)
        if mag is not None:
            for key, ss, nn in (("mag", mag_s, mag_n), ("magx", magx_s, magx_n)):
                for w, win in R.WINDOWS.items():
                    res[f"{key}_{w}"] = R.band_weighted(t_rel, ss, nn, win, (2.0, 15.0))
        pre_n, pre_o = res.get("gran_near_pre"), res.get("gran_outer_pre")
        res["inflow_ring_pre"] = bool(np.isfinite(pre_n) and np.isfinite(pre_o)
                                      and pre_n < 0 < pre_o)
    (out / "quicklook").mkdir(exist_ok=True)
    with open(out / "quicklook" / "radial_test.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    plot_event(out / "quicklook" / "radial_test.png", eid, t_h, gmap, mag, mag_x,
               t_pen, on["t_moat_h"], d["pf_lit"],
               (mag_s, mag_n, magx_s, magx_n) if mag is not None else None)
    print(f"{eid}: penumbra onset {t_pen:.1f} h | granular v_r near edge (0-3 Mm) "
          f"pre {res.get('gran_near_pre', np.nan):+.2f}, post {res.get('gran_near_post', np.nan):+.2f} km/s; "
          f"5-12 Mm pre {res.get('gran_outer_pre', np.nan):+.2f}, post {res.get('gran_outer_post', np.nan):+.2f}"
          + (f" | magnetic drift 2-15 Mm pre {res.get('mag_pre', np.nan):+.2f}, "
             f"post {res.get('mag_post', np.nan):+.2f}" if mag is not None else
             " | no mmf_tracks.npz (run mmf_onset.py for test 2)"))
    return res


def _map(ax, x, m, title, t_marks, pf=None):
    im = ax.pcolormesh(x, R.D_EDGES, m.T, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                       shading="flat")
    ax.axhline(0, color="k", lw=0.8)
    for y in (R.NEAR_MM[1], R.OUTER_MM[0], R.OUTER_MM[1]):
        ax.axhline(y, color="k", lw=0.4, ls=":")
    for tt, col, ls in t_marks:
        if np.isfinite(tt):
            ax.axvline(tt, color=col, ls=ls, lw=1.3)
    if pf is not None and np.isfinite(pf[0]):
        ax.axvspan(pf[0], pf[1] if np.isfinite(pf[1]) else pf[0] + 0.3,
                   ymin=0.97, ymax=1.0, color="green")
    ax.set_ylabel("distance from spot edge [Mm]")
    ax.set_title(title, fontsize=9)
    return im


def plot_event(png, eid, t_h, gmap, mag, mag_x, t_pen, t_moat, pf, magsc=None):
    edges_t = np.r_[t_h - 0.5, t_h[-1] + 0.5]
    rows = 3 if mag is not None else 2
    fig, axs = plt.subplots(rows, 1, figsize=(12, 3.4 * rows + 0.6), sharex=False)
    marks = [(t_pen, "tab:orange", "--"), (t_moat, "tab:blue", "--")]
    im = _map(axs[0], edges_t, gmap, f"{eid} test 1: granular v_r (azimuthal mean; red = outflow)",
              marks, pf)
    fig.colorbar(im, ax=axs[0], pad=0.01, label="v$_r$ [km/s]")
    if mag is not None:
        im = _map(axs[1], edges_t, mag, "test 2: radial drift of long-lived (>= 1 h) magnetic "
                  "features, 30 min baselines", marks, pf)
        fig.colorbar(im, ax=axs[1], pad=0.01, label="drift [km/s]")
    for ax in axs[:-1]:
        ax.set_xlabel("hours since start")
    ax = axs[-1]
    if np.isfinite(t_pen):
        t_rel = t_h - t_pen
        for w, p in R.window_profiles(t_rel, gmap).items():
            ax.plot(R.D_MID, p, "-", color=WCOL[w], lw=1.8,
                    label=f"granular, {w} ({R.WINDOWS[w][0]:+.0f}..{R.WINDOWS[w][1]:+.0f} h)")
        if magsc is not None:
            ms, mn, xs, xn = magsc
            for w, p in R.window_weighted(t_rel, ms, mn).items():
                ax.plot(R.D_MID, p, "--", color=WCOL[w], lw=1.1, label=f"magnetic, {w}")
            for w, p in R.window_weighted(t_rel, xs, xn).items():
                ax.plot(R.D_MID, p, ":", color=WCOL[w], lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvspan(*R.NEAR_MM, color="0.9", zorder=0)
    ax.set(xlabel="distance from spot edge [Mm]", ylabel="v$_r$ / drift [km/s]",
           ylim=(-0.6, 0.6), title="profiles in windows around the penumbra onset "
           "(magnetic: count-weighted, bins with >= 10 measurements; dotted: without "
           "outward MMFs; grey: 0-3 Mm)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)


def summary(cfg, ids):
    rd = cfg["data_root"] / "sample_results"
    rd.mkdir(exist_ok=True)
    used = {}
    f = rd / "onsets.csv"
    if f.exists():
        used = {r["event"]: r["used"] == "True" for r in csv.DictReader(open(f))}
    res, gm, mm, mx, trs, msn = [], [], [], [], [], []
    for eid in ids:
        out = event_dir(cfg, eid)
        j, z = out / "quicklook" / "radial_test.json", out / "radial_test.npz"
        if not (j.exists() and z.exists()):
            continue
        r = json.load(open(j))
        r["used"] = used.get(eid, False)
        res.append(r)
        if r["used"] and np.isfinite(r["t_pen_h"]):
            dz = np.load(z)
            trs.append(dz["t_h"] - r["t_pen_h"])
            gm.append(dz["granular"])
            mm.append(dz["magnetic"])
            mx.append(dz["magnetic_no_mmf"])
            if "mag_sums" in dz.files:
                msn.append((dz["t_h"] - r["t_pen_h"], dz["mag_sums"], dz["mag_counts"]))
    if not res:
        return
    buf = io.StringIO()

    def fm(x):
        return "   -  " if x is None or not np.isfinite(x) else f"{x:+.2f} "
    print("Radial flow structure around the penumbra onset [km/s]; windows pre "
          "-14..-2 h, post +6..+18 h", file=buf)
    print("granular v_r near the edge (0-3 Mm) and farther out (5-12 Mm); magnetic "
          "drift 2-15 Mm (all long-lived features / without outward MMFs)\n", file=buf)
    print(f"{'event':10s} used  near pre  outer pre | near post outer post | "
          f"mag pre  mag post | noMMF pre noMMF post | inflow ring pre?", file=buf)
    for r in res:
        print(f"{r['event']:10s} {'yes' if r['used'] else 'no ':4s} "
              f"{fm(r.get('gran_near_pre'))}   {fm(r.get('gran_outer_pre'))}   | "
              f"{fm(r.get('gran_near_post'))}   {fm(r.get('gran_outer_post'))}    | "
              f"{fm(r.get('mag_pre'))}  {fm(r.get('mag_post'))}  | "
              f"{fm(r.get('magx_pre'))}    {fm(r.get('magx_post'))}    | "
              f"{'yes' if r.get('inflow_ring_pre') else 'no'}", file=buf)
    u = [r for r in res if r["used"]]
    if u:
        n_ring = sum(bool(r.get("inflow_ring_pre")) for r in u)
        print(f"\nused events: inflow at the edge with outflow farther out before the "
              f"onset in {n_ring} of {len(u)}", file=buf)
        for key, lab in (("gran_near_pre", "granular 0-3 Mm, pre"),
                         ("gran_near_post", "granular 0-3 Mm, post"),
                         ("gran_outer_pre", "granular 5-12 Mm, pre"),
                         ("gran_outer_post", "granular 5-12 Mm, post"),
                         ("mag_pre", "magnetic drift, pre"), ("mag_post", "magnetic drift, post"),
                         ("magx_pre", "magnetic drift w/o MMFs, pre"),
                         ("magx_post", "magnetic drift w/o MMFs, post")):
            v = [r.get(key) for r in u if r.get(key) is not None and np.isfinite(r.get(key))]
            if v:
                print(f"  {lab:32s} median {np.median(v):+.2f} km/s "
                      f"(range {min(v):+.2f} .. {max(v):+.2f}, n = {len(v)})", file=buf)
    txt = buf.getvalue()
    print("\n" + txt)
    (rd / "radial_test_summary.txt").write_text(txt)

    if not gm:
        return
    panels = [("test 1: granular v_r", gm), ("test 2: magnetic drift (all long-lived)", mm),
              ("test 2: magnetic drift without outward MMFs", mx)]
    fig, axs = plt.subplots(len(panels) + 1, 1, figsize=(11, 3.2 * (len(panels) + 1)))
    stacked = {}
    for ax, (title, maps) in zip(axs, panels):
        grid, mean, cnt = R.stack(maps, trs)
        stacked[title] = (grid, mean)
        edges = np.r_[grid - 0.5, grid[-1] + 0.5]
        mean = np.where(cnt >= max(2, len(maps) // 2), mean, np.nan)
        im = _map(ax, edges, mean, f"{title}: mean of {len(maps)} used events, aligned on "
                  "penumbra onset (cells with < half the events blank)", [(0, "tab:orange", "--")])
        fig.colorbar(im, ax=ax, pad=0.01)
        ax.set_xlabel("hours from penumbra onset")
    ax = axs[-1]
    g_grid, g_mean = stacked[panels[0][0]]
    for w, (a, b) in R.WINDOWS.items():
        sel = (g_grid >= a) & (g_grid < b)
        with np.errstate(all="ignore"):
            p = np.nanmean(g_mean[sel], axis=0)
        ax.plot(R.D_MID, p, "-", color=WCOL[w], lw=1.8,
                label=f"granular, {w} ({a:+.0f}..{b:+.0f} h)")
        # magnetic: all events' measurements in the window pooled
        s_ = sum(sm[(tr >= a) & (tr < b)].sum(axis=0) for tr, sm, _ in msn) if msn else None
        c_ = sum(cn[(tr >= a) & (tr < b)].sum(axis=0) for tr, _, cn in msn) if msn else None
        if msn:
            with np.errstate(invalid="ignore", divide="ignore"):
                ax.plot(R.D_MID, np.where(c_ >= 10, s_ / c_, np.nan), "--", color=WCOL[w],
                        lw=1.1, label=f"magnetic (pooled), {w}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvspan(*R.NEAR_MM, color="0.9", zorder=0)
    ax.set(xlabel="distance from spot edge [Mm]", ylabel="km/s", ylim=(-0.6, 0.6),
           title="stacked profiles in windows around the penumbra onset")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rd / "radial_test_stack.png", dpi=120)
    plt.close(fig)
    print(f"-> {rd / 'radial_test_summary.txt'}, radial_test_stack.png")


def main():
    ap = argparse.ArgumentParser()
    add_event_args(ap)
    args = ap.parse_args()
    cfg = load_config()
    ids = resolve_events(args, cfg, ap)
    code = run_batch(lambda e: run_event(cfg, e), ids, out_name="radial_test.png",
                     cfg=cfg, skip_existing=args.skip_existing)
    if len(ids) > 1:
        summary(cfg, ids)
    sys.exit(code)


if __name__ == "__main__":
    main()
