#!/usr/bin/env python
"""Penumbra vs moat onset per azimuthal sector of the tracked spot.

For spots that form their penumbra on one side only (AR13010: southern
sector). Tracks the same spot on the same epochs as onset_analysis
(moatflow.analysis.eventdata), splits penumbral area and annulus outflow
(granulation and MMF) into W / N / E / S quadrants, and applies the
whole-spot onset rule to each quadrant. Run onset_analysis.py first.

Usage: python scripts/sector_analysis.py AR13010 [--seed-h 48]

Writes data/<event>/sector_series.npz, quicklook/sector_onsets.json and
quicklook/sector_comparison.png:
  top     time-azimuth maps (30 deg sectors) of penumbral area and of
          annulus outflow -- where around the spot each appears, and when
  bottom  one panel per quadrant: penumbral area and outflow curves with
          that quadrant's onsets
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moatflow.analysis.eventdata import load_tracked_event
from moatflow.analysis.onsets import analyse, smooth
from moatflow.analysis.sectors import sector_onsets, sector_series
from moatflow.config import load_config


def fmt(x, sign=False):
    if x is None or not np.isfinite(x):
        return "   -"
    return f"{x:+5.1f}" if sign else f"{x:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_id")
    ap.add_argument("--seed-h", type=float, default=None,
                    help="tracking seed [h]; default: the seed "
                         "onset_analysis used")
    args = ap.parse_args()

    cfg = load_config()
    d = load_tracked_event(cfg, args.event_id, seed_h=args.seed_h)
    out, t, tr = d["out"], d["t_h"], d["tr"]
    (out / "quicklook").mkdir(exist_ok=True)
    print(f"tracked {tr['valid'].sum()}/{len(t)} epochs "
          f"(seed {d['seed_h']:.0f} h)")

    ser = sector_series(d["frames"], tr, d["VX"], d["VY"], d["VXm"],
                        d["VYm"])
    res = sector_onsets(t, ser, tr["valid"])
    g = np.isfinite(ser["bg_vx"])
    if g.any():
        print(f"background (quiet-Sun) flow removed before the split: "
              f"v_x {ser['bg_vx'][g][0] * 1000:+.0f} -> "
              f"{ser['bg_vx'][g][-1] * 1000:+.0f} m/s, v_y "
              f"{ser['bg_vy'][g][0] * 1000:+.0f} -> "
              f"{ser['bg_vy'][g][-1] * 1000:+.0f} m/s (first -> last epoch)")
    whole = analyse(np.load(out / "onset_series.npz"))
    pf0, pf1 = d["pf_lit"]

    np.savez(out / "sector_series.npz", t_h=t, valid=tr["valid"],
             names=np.array(ser["names"]), az_deg=ser["az_deg"],
             **{k: ser[k] for k in ("a_pen", "a_umb", "v", "v_mmf",
                                    "frac_bad", "az_v", "az_pen", "bg_vx",
                                    "bg_vy", "bg_vx_mmf", "bg_vy_mmf")})
    keep = {nm: {k: (v if not isinstance(v, (np.floating, np.bool_))
                     else v.item())
                 for k, v in r.items() if not k.startswith("_")}
            for nm, r in res.items()}
    keep["whole_spot"] = {"t_pen_h": whole["t_pen_h"],
                          "t_moat_h": whole["t_moat_h"],
                          "t_mmf_h": whole["t_mmf_h"],
                          "lag_moat_h": whole["lag_moat_h"]}
    with open(out / "quicklook" / "sector_onsets.json", "w") as f:
        json.dump(keep, f, indent=2, default=float)

    lit = (f"{pf0:.1f} - {pf1:.1f} h" if np.isfinite(pf1)
           else f"starts {pf0:.1f} h")
    print(f"literature formation: {lit}; whole spot: penumbra "
          f"{fmt(whole['t_pen_h'])} h, moat {fmt(whole['t_moat_h'])} h, "
          f"MMF {fmt(whole['t_mmf_h'])} h")
    print("sector  pen base->mature [Mm^2]   pen on  moat on  MMF on   "
          "lag  lagMMF  moat mature  holds  contam")
    for nm, r in res.items():
        grow = "" if r["forms_penumbra"] else "  (no growth)"
        print(f"  {nm}    {r['pen_baseline_Mm2']:6.1f} -> "
              f"{r['pen_mature_Mm2']:6.1f}{grow:13s} "
              f"{fmt(r['t_pen_h'])}   {fmt(r['t_moat_h'])}   "
              f"{fmt(r['t_mmf_h'])}  {fmt(r['lag_moat_h'], True)}  "
              f"{fmt(r['lag_mmf_h'], True)}   "
              f"{r['moat_mature_km_s']:5.2f} km/s   "
              f"{'yes' if r['moat_holds'] is True else 'NO ' if r['moat_holds'] is False else ' - '}"
              f"   {r['contamination_at_moat_onset'] * 100:4.0f} %")

    # ---- figure
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.1, 1.1, 1, 1],
                          hspace=0.45, wspace=0.32)
    az = ser["az_deg"]
    ext = [t[0] - 0.5, t[-1] + 0.5, 0, 360]
    for row, (key, title, kw) in enumerate((
            ("az_pen", "penumbral area per 30 deg sector [Mm$^2$]",
             dict(cmap="Oranges", vmin=0)),
            ("az_v", "annulus outflow per 30 deg sector, quiet-Sun "
             "background flow removed [km/s]",
             dict(cmap="RdBu_r", vmin=-0.4, vmax=0.4)))):
        ax = fig.add_subplot(gs[row, :])
        m = np.where(tr["valid"][:, None], ser[key], np.nan).T
        im = ax.imshow(m, aspect="auto", origin="lower", extent=ext,
                       interpolation="nearest", **kw)
        fig.colorbar(im, ax=ax, pad=0.01)
        ax.set_yticks([0, 90, 180, 270, 360],
                      ["W", "N", "E", "S", "W"])
        ax.set_ylabel("azimuth")
        for tt, col in ((whole["t_pen_h"], "tab:orange"),
                        (whole["t_moat_h"], "tab:blue")):
            if np.isfinite(tt):
                ax.axvline(tt, color=col, ls="--", lw=1.2)
        for x in (pf0, pf1):
            if np.isfinite(x):
                ax.axvline(x, color="green", lw=1.4)
        ax.set_title(title + "   (dashed: whole-spot onsets, green: "
                     "literature formation)", fontsize=9)
    for j, nm in enumerate(res):
        r = res[nm]
        ax = fig.add_subplot(gs[2 + j // 2, j % 2])
        ax2 = ax.twinx()
        a = np.where(tr["valid"], ser["a_pen"][:, j], np.nan)
        ax2.plot(t, smooth(a), color="tab:orange", lw=1.4)
        ax2.set_ylabel("penumbra [Mm$^2$]", color="tab:orange", fontsize=8)
        v = np.where(tr["valid"], ser["v"][:, j], np.nan)
        ax.plot(t, smooth(v), color="tab:blue", lw=1.4, label="moat")
        vm = np.where(tr["valid"], ser["v_mmf"][:, j], np.nan)
        if np.isfinite(vm).any():
            ax.plot(t, smooth(vm), color="tab:purple", lw=0.9, ls=":",
                    label="MMF")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-0.25, 0.5)
        ax.set_ylabel("v$_r$ [km/s]", color="tab:blue", fontsize=8)
        if np.isfinite(pf0):
            ax.axvspan(pf0, pf1 if np.isfinite(pf1) else pf0 + 0.3,
                       color="green", alpha=0.12)
        for tt, col in ((r["t_pen_h"], "tab:orange"),
                        (r["t_moat_h"], "tab:blue"),
                        (r["t_mmf_h"], "tab:purple")):
            if np.isfinite(tt):
                ax.axvline(tt, color=col, ls="--", lw=1.1)
        lag = (f"lag {r['lag_moat_h']:+.1f} h"
               if np.isfinite(r["lag_moat_h"]) else
               "no penumbra growth" if not r["forms_penumbra"] else
               "no moat onset")
        ax.set_title(f"{nm} quadrant: {lag}", fontsize=9)
        ax.set_xlim(t[0], t[-1])
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
        if j >= 2:
            ax.set_xlabel("hours since start")
    fig.suptitle(f"{args.event_id}: penumbra vs moat onset by sector "
                 "(0 deg = west, 90 deg = north)", y=0.995)
    png = out / "quicklook" / "sector_comparison.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
