#!/usr/bin/env python
"""Sample-level result: penumbra onset vs moat onset for every event.

Reads data/<EV>/onset_series.npz (written by onset_analysis.py) and, when
present, quicklook/verify_flct.json; applies the onset rules of
moatflow.analysis.onsets (identical for both curves); writes to
<data_root>/sample_results/:

  onsets.csv        one row per event: onset times (h since the first cube
                    frame, and TAI), lag, threshold sweep, quality numbers,
                    flags, and whether the event is used
  summary.txt       the printed table and the sample statistics
  events_grid.png   every event's two curves, normalised so that both
                    thresholds sit at the same height, with both onsets
                    marked -- check every pick by eye before trusting
                    the table
  superposed.png    the used events aligned on their penumbra onset, with
                    the median of each curve
  lags.png          lag per event for the threshold fractions of the sweep

An event is USED unless a blocking flag fires (no measurable onset, no
pore phase, onset inside the baseline window, moat on from the start), or
unless the catalog entry sets `onset_use: false`. `onset_use: true` forces
an event in despite its flags; add `onset_note:` to record why.

Usage:
  python scripts/sample_analysis.py
  python scripts/sample_analysis.py AR11490 AR11630 --out /path/to/dir
"""

import argparse
import csv
import io
import json
import warnings
from contextlib import redirect_stdout
from datetime import timedelta
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moatflow.analysis.onsets import FRACTION, FRACTION_SWEEP, analyse
from moatflow.catalog import load_events
from moatflow.config import load_config
from moatflow.viz import parse_datetimes

SIMULTANEOUS_H = 2.0      # |lag| below this: same epoch within resolution
SWEEP_KEYS = [f"lag_f{int(round(f * 100)):02d}_h" for f in FRACTION_SWEEP]


def window_start(d, event):
    """Time of the first cube frame, which t_h in onset_series counts from."""
    cube = d / "cube_hmi.Ic_45s_continuum.h5"
    if cube.exists():
        try:
            with h5py.File(cube, "r") as h5:
                t0 = h5["t_obs"][0]
            return parse_datetimes([t0.decode() if isinstance(t0, bytes)
                                    else t0])[0]
        except Exception:
            pass
    if event and event.get("t_start"):
        return parse_datetimes([event["t_start"]])[0]
    return None


def load_event(d, eid, event):
    npz = d / "onset_series.npz"
    if not npz.exists():
        return None
    vfile = d / "quicklook" / "verify_flct.json"
    verify = json.loads(vfile.read_text()) if vfile.exists() else None
    r = analyse(np.load(npz, allow_pickle=False), verify=verify)
    r["event"] = eid
    r["t0"] = window_start(d, event)

    override = event.get("onset_use") if event else None
    if override is None:
        r["used"] = not r["blocking"]
        r["use_reason"] = "; ".join(r["blocking"]) if r["blocking"] else "ok"
    else:
        r["used"] = bool(override)
        r["use_reason"] = ("catalog onset_use: " + str(override).lower()
                           + (f" ({event['onset_note']})"
                              if event.get("onset_note") else ""))
    r["has_verify"] = verify is not None
    return r


def fmt(x, nd=1, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def tai(r, key):
    if r["t0"] is None or not np.isfinite(r[key]):
        return ""
    return (r["t0"] + timedelta(hours=r[key])).strftime("%Y-%m-%d %H:%M")


def write_csv(rows, path):
    cols = ["event", "used", "use_reason", "lit_start_h", "lit_end_h",
            "t_first_valid_h", "t_pen_h", "t_pen_first_departure_h",
            "t_moat_h", "t_mmf_h", "lag_moat_h", "lag_mmf_h", *SWEEP_KEYS,
            "t_moat_first_h", "moat_first_thr_km_s", "lag_first_h",
            "pre_onset_pen_h", "pre_onset_moat_h", "pen_holds", "moat_holds",
            "contamination_at_moat_onset",
            "pen_baseline_Mm2", "pen_mature_Mm2", "pen_baseline_frac",
            "moat_mature_km_s", "ring_snr", "contamination_median",
            "contamination_max", "doppler_r", "doppler_amp",
            "shrink_sun_shift_m_s"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "window_start_tai"] + cols[1:] +
                    ["t_pen_tai", "t_moat_tai", "blocking", "warnings"])
        for r in rows:
            vals = []
            for c in cols[1:]:
                v = r[c]
                vals.append(round(v, 3) if isinstance(v, float) and
                            np.isfinite(v) else ("" if isinstance(v, float)
                                                 else v))
            w.writerow([r["event"],
                        r["t0"].strftime("%Y-%m-%d %H:%M:%S") if r["t0"]
                        else ""] + vals +
                       [tai(r, "t_pen_h"), tai(r, "t_moat_h"),
                        " | ".join(r["blocking"]), " | ".join(r["warnings"])])


def report(rows, missing):
    print(f"Onset rule: both curves smoothed over 3 h; threshold at "
          f"{FRACTION:.0%} of the way from 'none' to the mature level "
          f"(penumbra: pore baseline -> 90th pct; moat: 0 -> 90th pct); "
          f"onset = first of 3 consecutive hours above.")
    print(f"Lag = moat onset - penumbra onset; positive = moat AFTER "
          f"penumbra. Times in h since the first cube frame.")
    print("lag1st = the same for FIRST APPEARANCE (penumbra: baseline + "
          "max(3 sigma, 20 Mm^2); moat: 3 sigma of its noise above 0, "
          ">= 0.05 km/s).\n")
    hdr = (f"{'event':8s} {'use':4s} {'lit':>11s} {'pen':>6s} {'(1st)':>6s} "
           f"{'moat':>6s} {'MMF':>6s} {'lag':>6s} {'lag1st':>6s}  "
           + " ".join(f"{k[5:7]+'%':>5s}" for k in SWEEP_KEYS)
           + f"  {'SNR':>4s} {'Dopp':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lit = f"{fmt(r['lit_start_h'],0)}-{fmt(r['lit_end_h'],0)}"
        print(f"{r['event']:8s} {'yes' if r['used'] else 'NO':4s} {lit:>11s} "
              f"{fmt(r['t_pen_h']):>6s} {fmt(r['t_pen_first_departure_h']):>6s} "
              f"{fmt(r['t_moat_h']):>6s} {fmt(r['t_mmf_h']):>6s} "
              f"{fmt(r['lag_moat_h'], sign=True):>6s} "
              f"{fmt(r['lag_first_h'], sign=True):>6s}  "
              + " ".join(f"{fmt(r[k], 0, True):>5s}" for k in SWEEP_KEYS)
              + f"  {fmt(r['ring_snr']):>4s} {fmt(r['doppler_r'], 2):>5s}")
    print()
    for r in rows:
        notes = ([f"EXCLUDED: {r['use_reason']}"] if not r["used"] else []) \
            + ([f"forced in: {r['use_reason']}"]
               if r["used"] and r["blocking"] else []) + r["warnings"] \
            + ([] if r["has_verify"] else ["no verify_flct.json yet"])
        if notes:
            print(f"  {r['event']}: " + "; ".join(notes))
    if missing:
        print(f"\n  not analysed yet (no onset_series.npz): {', '.join(missing)}")

    used = [r for r in rows if r["used"] and np.isfinite(r["lag_moat_h"])]
    print(f"\nSample: {len(used)} event(s) used")
    if not used:
        return
    lags = np.array([r["lag_moat_h"] for r in used])
    after = int((lags > SIMULTANEOUS_H).sum())
    before = int((lags < -SIMULTANEOUS_H).sum())
    print(f"  lag moat - penumbra: median {np.median(lags):+.1f} h, "
          f"range {lags.min():+.1f} .. {lags.max():+.1f} h")
    print(f"  moat after penumbra: {after}, within +-{SIMULTANEOUS_H:.0f} h: "
          f"{len(lags) - after - before}, before: {before}")
    first = np.array([r["lag_first_h"] for r in used
                      if np.isfinite(r["lag_first_h"])])
    if len(first):
        fa = int((first > SIMULTANEOUS_H).sum())
        fb = int((first < -SIMULTANEOUS_H).sum())
        print(f"  first appearance: median {np.median(first):+.1f} h, range "
              f"{first.min():+.1f} .. {first.max():+.1f} h; after {fa}, "
              f"within {len(first) - fa - fb}, before {fb}")
    for k in SWEEP_KEYS:
        s = np.array([r[k] for r in used if np.isfinite(r[k])])
        if len(s):
            print(f"  f = {k[5:7]}%: median lag {np.median(s):+.1f} h "
                  f"(n = {len(s)})")
    mmf = np.array([r["lag_mmf_h"] for r in used
                    if np.isfinite(r["lag_mmf_h"])])
    if len(mmf):
        print(f"  MMF lag (secondary; MMFs exist only once a penumbra "
              f"sheds them): median {np.median(mmf):+.1f} h (n = {len(mmf)})")


def norm(curve):
    """Map a curve so that 'none' -> 0 and its mature level -> 1."""
    span = curve["mature"] - curve["zero"]
    return (curve["smooth"] - curve["zero"]) / span if span else \
        np.full_like(curve["smooth"], np.nan)


def plot_grid(rows, path):
    n = len(rows)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.3 * nrow),
                             squeeze=False)
    for ax, r in zip(axes.flat, rows):
        c = r["_curves"]
        t = c["t"]
        ax.plot(t, norm(c["pen"]), color="tab:orange", lw=1.6,
                label="penumbral area")
        ax.plot(t, norm(c["moat"]), color="tab:blue", lw=1.6,
                label="moat outflow")
        if c["mmf"] is not None:
            ax.plot(t, norm(c["mmf"]), color="tab:purple", lw=0.9, ls=":",
                    label="MMF outflow")
        lo, hi = r["lit_start_h"], r["lit_end_h"]
        if np.isfinite(lo) and np.isfinite(hi) and hi > t[0] and lo < t[-1]:
            ax.axvspan(max(lo, t[0]), min(hi, t[-1]), color="green",
                       alpha=0.10, label="literature formation")
        elif np.isfinite(lo) and not np.isfinite(hi) and t[0] <= lo <= t[-1]:
            ax.axvline(lo, color="green", lw=1.2, ls="-.",
                       label="literature formation start")
        elif np.isfinite(lo):
            ax.text(0.98, 0.04, "literature interval outside the data",
                    transform=ax.transAxes, ha="right", fontsize=7,
                    color="firebrick")
        ax.axhline(FRACTION, color="gray", lw=0.6, ls="--")
        ax.axhline(0, color="k", lw=0.4)
        for key, col in (("t_pen_h", "tab:orange"), ("t_moat_h", "tab:blue")):
            if np.isfinite(r[key]):
                ax.axvline(r[key], color=col, lw=1.0, ls="--")
        for key, col in (("t_pen_first_departure_h", "tab:orange"),
                         ("t_moat_first_h", "tab:blue")):
            if np.isfinite(r[key]):
                ax.axvline(r[key], color=col, lw=0.8, ls=":")
        ax.set_xlim(t[0] - 1, t[-1] + 1)
        state = "used" if r["used"] else "EXCLUDED"
        ax.set_title(f"{r['event']}  lag {fmt(r['lag_moat_h'], sign=True)} h"
                     f" (1st {fmt(r['lag_first_h'], sign=True)})  [{state}]",
                     fontsize=10,
                     color="k" if r["used"] else "firebrick")
        ax.set_ylim(-0.5, 1.4)
        ax.set_xlabel("h since first cube frame", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Each curve scaled so 0 = none, 1 = mature level; dashed "
                 f"grey = threshold ({FRACTION:.0%}); dashed verticals = "
                 "onsets (40 % rule), dotted = first appearance",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_superposed(rows, path):
    used = [r for r in rows if r["used"] and np.isfinite(r["t_pen_h"])]
    if not used:
        return
    grid = np.arange(-40, 50.5, 1.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    stacks = {"pen": [], "moat": []}
    for r in used:
        c = r["_curves"]
        x = c["t"] - r["t_pen_h"]
        for key, col in (("pen", "tab:orange"), ("moat", "tab:blue")):
            y = norm(c[key])
            ax.plot(x, y, color=col, lw=0.7, alpha=0.35)
            ok = np.isfinite(y)
            g = np.full_like(grid, np.nan)
            inside = (grid >= x[ok].min()) & (grid <= x[ok].max()) \
                if ok.any() else np.zeros_like(grid, bool)
            g[inside] = np.interp(grid[inside], x[ok], y[ok])
            stacks[key].append(g)
    for key, col, lab in (("pen", "tab:orange", "penumbral area"),
                          ("moat", "tab:blue", "moat outflow")):
        s = np.array(stacks[key])
        n_ok = np.isfinite(s).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
            med = np.where(n_ok >= max(2, len(used) // 2),
                           np.nanmedian(s, axis=0), np.nan)
        ax.plot(grid, med, color=col, lw=2.6, label=f"{lab}, median")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(FRACTION, color="gray", lw=0.6, ls="--")
    ax.axhline(0, color="k", lw=0.4)
    ax.set(xlabel="hours since penumbra onset",
           ylabel="0 = none, 1 = mature level",
           title=f"{len(used)} events aligned on the penumbra onset "
                 "(thin: individual events)", ylim=(-0.6, 1.4))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_lags(rows, path):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.6))
    marks = ["v", "o", "^"]
    for i, r in enumerate(rows):
        col = "tab:blue" if r["used"] else "0.7"
        for k, m, f in zip(SWEEP_KEYS, marks, FRACTION_SWEEP):
            if np.isfinite(r[k]):
                ax.plot(r[k], i, m, color=col, ms=7 if f == FRACTION else 5,
                        label=f"f = {f:.0%}" if i == 0 else None)
        if np.isfinite(r["lag_first_h"]):
            ax.plot(r["lag_first_h"], i, "x",
                    color="tab:red" if r["used"] else "0.7", ms=7,
                    label="first appearance" if i == 0 else None)
    ax.axvspan(-SIMULTANEOUS_H, SIMULTANEOUS_H, color="0.9", zorder=0)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["event"] + ("" if r["used"] else " (excl.)")
                        for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="lag moat - penumbra [h]  (positive = moat after)",
           title="Lag per event and threshold fraction (grey: excluded)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="*", help="event ids (default: catalog)")
    ap.add_argument("--out", default=None,
                    help="output directory (default <data_root>/sample_results)")
    args = ap.parse_args()

    cfg = load_config()
    catalog = load_events()
    out = Path(args.out) if args.out else cfg["data_root"] / "sample_results"
    out.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for eid in (args.events or list(catalog)):
        r = load_event(cfg["data_root"] / eid, eid, catalog.get(eid))
        if r is None:
            missing.append(eid)
        else:
            rows.append(r)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(rows, missing)
    text = buf.getvalue()
    print(text)
    (out / "summary.txt").write_text(text)
    if rows:
        write_csv(rows, out / "onsets.csv")
        plot_grid(rows, out / "events_grid.png")
        plot_superposed(rows, out / "superposed.png")
        plot_lags(rows, out / "lags.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
