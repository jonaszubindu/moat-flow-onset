#!/usr/bin/env python
"""Follow-up statistics on the tracked spots (run after onset_analysis and
sector_analysis).

  1. formation wedge -- the 30 deg sector where the penumbra appears
     first, widened to a 90 deg wedge centred on it (independent of the
     fixed quadrant boundaries): its moat onset vs its penumbra onset,
     and the same for the opposite wedge.
  2. moat vs penumbra -- per event, correlation of the smoothed whole-spot
     moat curve with penumbral area (size) and with its growth rate,
     including the time shift that maximises the correlation; across
     events, mature moat vs mature penumbral area; per quadrant, whether
     a weak quadrant moat goes with a small quadrant penumbra.
  3. bias test -- west-minus-east quadrant moat asymmetry (background
     flow already removed) in mature epochs east vs west of the central
     meridian: a residual disk-centre LCT bias flips sign at the
     meridian, a physical asymmetry does not; plus a regression of the
     asymmetry on the measured background flow.
  4. leading or trailing -- polarity of the tracked umbra (magnetogram at
     the seed epoch), its position relative to the opposite-polarity flux
     in the patch, and Hale's law for its hemisphere and cycle.

Usage: python scripts/followup_stats.py [EVENT ...]   (default: every
event with onset_series.npz). Writes data/followup/report.txt and
data/followup/{correlation,bias_test}.png.
"""

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from moatflow.analysis.eventdata import load_tracked_event
from moatflow.analysis.onsets import (MATURE_PCT, analyse, moat_onset,
                                      penumbra_onset, smooth)
from moatflow.analysis.penumbra import segment_frame
from moatflow.analysis.spottrack import PX_MM
from moatflow.analysis.verify import synodic_rate_deg_per_h
from moatflow.catalog import load_events
from moatflow.config import event_dir, load_config
from moatflow.cubes import load_cube
from moatflow.viz import parse_datetimes

SECTOR_GROWTH_MM2 = 5.0        # a 30 deg sector "forms penumbra" above this
MAX_SHIFT_H = 24               # cross-correlation range
MERIDIAN_GAP_DEG = 5.0         # epochs within +-5 deg of CM are left out
MATURE_AFTER_H = 6.0           # mature = 6 h after the later whole-spot onset
B_STRONG = 100.0               # G, flux counted for polarity centroids


def lon_series(event, t0, t_h):
    lat = float(event["lat_ref"])
    t_ref_h = (parse_datetimes([event["t_ref"]])[0] - t0).total_seconds() / 3600
    return float(event["lon_ref"]) + synodic_rate_deg_per_h(lat) * (t_h - t_ref_h)


def window_t0(out):
    import h5py
    with h5py.File(out / "cube_hmi.Ic_45s_continuum.h5") as f:
        t = f["t_obs"][0]
    return parse_datetimes([t.decode() if isinstance(t, bytes) else t])[0]


# ---------------------------------------------------------------- 1
def formation_wedge(ser):
    t, valid = ser["t_h"], ser["valid"]
    az_pen, az_v = ser["az_pen"], ser["az_v"]
    n = az_pen.shape[1]
    first = []
    for j in range(n):
        p = penumbra_onset(t, az_pen[:, j], valid)
        if np.isfinite(p["mature"]) and p["mature"] - p["zero"] > SECTOR_GROWTH_MM2:
            first.append((p["t"], j))
    first = [f for f in first if np.isfinite(f[0])]
    if not first:
        return None
    t_first, j0 = min(first)

    def wedge(jc):
        js = [(jc + d) % n for d in (-1, 0, 1)]
        a = np.nansum(az_pen[:, js], axis=1)
        with np.errstate(all="ignore"):
            v = np.nanmean(az_v[:, js], axis=1)
        pen = penumbra_onset(t, a, valid)
        moat = moat_onset(t, v, valid)
        grows = np.isfinite(pen["mature"]) and pen["mature"] - pen["zero"] > \
            3 * SECTOR_GROWTH_MM2
        tp = pen["t"] if grows else np.nan
        return {"az": float(ser["az_deg"][jc % n]), "t_pen": tp,
                "t_moat": moat["t"], "lag": moat["t"] - tp,
                "moat_mature": moat["mature"],
                "pen_growth": pen["mature"] - pen["zero"]}
    return {"first_sector_az": float(ser["az_deg"][j0]),
            "t_first_sector": t_first,
            "formation": wedge(j0), "opposite": wedge(j0 + n // 2)}


# ---------------------------------------------------------------- 2
def lagged_corr(x, y, max_shift):
    """r(tau) of y(t + tau) against x(t); tau > 0: y lags x."""
    out = []
    for tau in range(-max_shift, max_shift + 1):
        if tau >= 0:
            a, b = x[:len(x) - tau or None], y[tau:]
        else:
            a, b = x[-tau:], y[:len(y) + tau]
        g = np.isfinite(a) & np.isfinite(b)
        out.append(stats.pearsonr(a[g], b[g])[0] if g.sum() > 10 else np.nan)
    return np.arange(-max_shift, max_shift + 1), np.array(out)


def moat_vs_penumbra(npz):
    t = npz["t_h"]
    valid = npz["valid"].astype(bool)
    a = smooth(np.where(valid, npz["a_pen"], np.nan))
    v = smooth(np.where(valid, npz["v_gran"], np.nan))
    da = np.gradient(a, t)
    g = np.isfinite(a) & np.isfinite(v)
    r_size = stats.pearsonr(a[g], v[g])[0] if g.sum() > 10 else np.nan
    rho_size = stats.spearmanr(a[g], v[g])[0] if g.sum() > 10 else np.nan
    gd = g & np.isfinite(da)
    r_rate = stats.pearsonr(da[gd], v[gd])[0] if gd.sum() > 10 else np.nan
    dt = float(np.median(np.diff(t)))
    taus, rt = lagged_corr(a, v, int(round(MAX_SHIFT_H / dt)))
    k = int(np.nanargmax(rt)) if np.isfinite(rt).any() else None
    return {"r_size": r_size, "rho_size": rho_size, "r_rate": r_rate,
            "tau_best_h": taus[k] * dt if k is not None else np.nan,
            "r_best": rt[k] if k is not None else np.nan,
            "taus_h": taus * dt, "r_tau": rt,
            "pen_mature": float(np.nanpercentile(a[g], MATURE_PCT)) if g.any() else np.nan,
            "moat_mature": float(np.nanpercentile(v[g], MATURE_PCT)) if g.any() else np.nan}


# ---------------------------------------------------------------- 3
def ew_asymmetry(ser, lon, t_mature):
    names = list(ser["names"])
    if "W" not in names or "E" not in names:
        return None
    w, e = names.index("W"), names.index("E")
    valid = ser["valid"].astype(bool)
    A = np.where(valid, ser["v"][:, w] - ser["v"][:, e], np.nan)
    mature = ser["t_h"] >= t_mature
    east = mature & (lon < -MERIDIAN_GAP_DEG) & np.isfinite(A)
    west = mature & (lon > MERIDIAN_GAP_DEG) & np.isfinite(A)

    def ms(sel):
        return ((float(np.mean(A[sel])), float(stats.sem(A[sel])), int(sel.sum()))
                if sel.sum() > 2 else (np.nan, np.nan, int(sel.sum())))
    return {"A": A, "lon": lon, "mature": mature, "bg_vx": ser["bg_vx"],
            "east": ms(east), "west": ms(west)}


# ---------------------------------------------------------------- 4
def hale_leading_sign(when, lat):
    """Leading-spot polarity expected from Hale's law."""
    cycle25 = when >= datetime(2019, 12, 1)
    north = lat >= 0
    # cycle 24: north leading negative, south positive; cycle 25 reversed
    return (+1 if north else -1) if cycle25 else (-1 if north else +1)


def leading_or_trailing(cfg, eid):
    out = event_dir(cfg, eid)
    mfile = out / "cube_hmi.M_45s_magnetogram.h5"
    if not mfile.exists():
        return {"note": "no magnetogram cube"}
    with redirect_stdout(io.StringIO()):
        d = load_tracked_event(cfg, eid, with_mmf=False)
    tr, k = d["tr"], d["tr"]["seed_idx"]
    frame = d["frames"][k]
    u, _ = segment_frame(frame)
    umb = u & tr["masks"][k]
    mg, t_mg, _ = load_cube(mfile)
    t_abs = d["t0"].timestamp() + d["t_mid"][k]
    tm = np.array([x.timestamp() for x in parse_datetimes(t_mg)])
    B = np.asarray(mg[int(np.argmin(np.abs(tm - t_abs)))], dtype=float)
    if B.shape != frame.shape or umb.sum() == 0:
        return {"note": "magnetogram grid differs from continuum or no umbra"}
    s = float(np.sign(np.nanmedian(B[umb])))
    b_spot = float(np.nanmedian(B[umb]))
    yy, xx = np.mgrid[0:B.shape[0], 0:B.shape[1]]
    opp = (s * B) < -B_STRONG
    same = (s * B) > B_STRONG
    phi_opp = float(np.abs(B[opp]).sum()) if opp.any() else 0.0
    phi_same = float(np.abs(B[same]).sum()) if same.any() else 0.0
    x_opp = float((np.abs(B[opp]) * xx[opp]).sum() / phi_opp) if phi_opp else np.nan
    y_opp = float((np.abs(B[opp]) * yy[opp]).sum() / phi_opp) if phi_opp else np.nan
    x_same = float((np.abs(B[same]) * xx[same]).sum() / phi_same) if phi_same else np.nan
    cy, cx = tr["center"][k]
    when = d["t0"]
    lat = float(d["event"]["lat_ref"])
    hale = hale_leading_sign(when, lat)
    az_opp = float(np.rad2deg(np.arctan2(y_opp - cy, x_opp - cx)) % 360)
    return {"b_umbra_G": b_spot, "polarity": int(s), "phi_opp_deg": az_opp,
            "dx_spot_minus_opp_Mm": (cx - x_opp) * PX_MM,
            "dx_same_minus_opp_Mm": (x_same - x_opp) * PX_MM,
            "flux_ratio_opp_over_same": phi_opp / phi_same if phi_same else np.nan,
            "hale_leading_polarity": hale, "lat": lat,
            "by_position": "leading" if cx > x_opp else "trailing",
            "by_hale": "leading" if s == hale else "trailing"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("event_ids", nargs="*")
    args = ap.parse_args()
    cfg = load_config()
    events = load_events()
    ids = args.event_ids or [e for e in events
                             if (event_dir(cfg, e) / "onset_series.npz").exists()]
    rep_dir = cfg["data_root"] / "followup"
    rep_dir.mkdir(exist_ok=True)
    buf = io.StringIO()

    def say(*a):
        print(*a)
        print(*a, file=buf)

    say(f"followup_stats  {datetime.now():%Y-%m-%d %H:%M}  events: {', '.join(ids)}")
    summ = {}
    sample_csv = cfg["data_root"] / "sample_results" / "onsets.csv"
    used = {}
    if sample_csv.exists():
        import csv
        used = {r["event"]: r["used"] == "True" for r in csv.DictReader(open(sample_csv))}

    # ---- 1
    say("\n1. FORMATION WEDGE (90 deg centred on the 30 deg sector whose penumbra appears first)")
    say("event      used  first-az  t_first | wedge: pen   moat    lag  | opposite: pen  moat   lag")
    for eid in ids:
        f = event_dir(cfg, eid) / "sector_series.npz"
        if not f.exists():
            continue
        ser = dict(np.load(f))
        fw = formation_wedge(ser)
        summ.setdefault(eid, {})["wedge"] = fw
        if fw is None:
            say(f"{eid:10s} no sector forms penumbra")
            continue
        a, o = fw["formation"], fw["opposite"]
        say(f"{eid:10s} {'yes' if used.get(eid) else 'no ':4s} {fw['first_sector_az']:6.0f}   "
            f"{fw['t_first_sector']:6.1f}  |  {a['t_pen']:5.1f}  {a['t_moat']:5.1f}  {a['lag']:+5.1f} "
            f" |  {o['t_pen']:5.1f}  {o['t_moat']:5.1f}  {o['lag']:+5.1f}")
    lags_f = [summ[e]["wedge"]["formation"]["lag"] for e in summ
              if summ[e].get("wedge") and used.get(e)]
    lags_f = [x for x in lags_f if np.isfinite(x)]
    if lags_f:
        say(f"  used events: formation-wedge lag median {np.median(lags_f):+.1f} h, "
            f"{sum(x > 2 for x in lags_f)} moat after / {sum(abs(x) <= 2 for x in lags_f)} within 2 h / "
            f"{sum(x < -2 for x in lags_f)} before (n = {len(lags_f)})")

    # ---- 2
    say("\n2. MOAT vs PENUMBRA (whole spot, 3 h smoothed, valid epochs)")
    say("event      used  r(size)  rho(size)  r(growth rate)  best shift  r(best)  pen mature  moat mature")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for eid in ids:
        npz = np.load(event_dir(cfg, eid) / "onset_series.npz")
        m = moat_vs_penumbra(npz)
        summ.setdefault(eid, {})["corr"] = m
        say(f"{eid:10s} {'yes' if used.get(eid) else 'no ':4s} {m['r_size']:+.2f}    {m['rho_size']:+.2f}"
            f"       {m['r_rate']:+.2f}          {m['tau_best_h']:+5.1f} h   {m['r_best']:+.2f}"
            f"    {m['pen_mature']:6.0f} Mm2  {m['moat_mature']:.2f} km/s")
        ax[0].plot(m["taus_h"], m["r_tau"], lw=2 if used.get(eid) else 0.8,
                   ls="-" if used.get(eid) else ":", label=eid)
        ax[1].scatter(m["pen_mature"], m["moat_mature"],
                      marker="o" if used.get(eid) else "x", s=40)
        ax[1].annotate(eid[2:], (m["pen_mature"], m["moat_mature"]), fontsize=7)
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set(xlabel="shift of moat vs penumbral area [h] (>0: moat lags)",
              ylabel="Pearson r", title="moat curve vs penumbral area, shifted")
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].set(xlabel="mature penumbral area [Mm$^2$]", ylabel="mature moat [km/s]",
              title="across events (o used, x excluded)")
    ax[1].grid(alpha=0.3)
    pm = [(summ[e]["corr"]["pen_mature"], summ[e]["corr"]["moat_mature"])
          for e in summ if "corr" in summ[e] and used.get(e)]
    if len(pm) >= 4:
        rho, p = stats.spearmanr(*zip(*pm))
        say(f"  across used events: Spearman rho(mature penumbra, mature moat) = {rho:+.2f} (p = {p:.2f}, n = {len(pm)})")
    used_corr = [summ[e]["corr"] for e in summ if "corr" in summ[e] and used.get(e)]
    if used_corr:
        say(f"  used events: median r(size) {np.median([c['r_size'] for c in used_corr]):+.2f}, "
            f"median r(growth rate) {np.median([c['r_rate'] for c in used_corr]):+.2f}, "
            f"median best shift {np.median([c['tau_best_h'] for c in used_corr]):+.1f} h")
    # quadrant level: does a weak quadrant moat go with a small quadrant penumbra?
    qp, qm = [], []
    for eid in ids:
        f = event_dir(cfg, eid) / "quicklook" / "sector_onsets.json"
        if not f.exists() or not used.get(eid):
            continue
        q = {k: v for k, v in json.load(open(f)).items() if k != "whole_spot"}
        pg = np.array([v["pen_mature_Mm2"] - v["pen_baseline_Mm2"] for v in q.values()])
        mv = np.array([v["moat_mature_km_s"] for v in q.values()])
        qp += list(pg / np.mean(pg))
        qm += list(mv / np.mean(mv))
        for (nm, v), p_, m_ in zip(q.items(), pg / np.mean(pg), mv / np.mean(mv)):
            ax[2].scatter(p_, m_, s=30, c={"W": "C0", "N": "C1", "E": "C3", "S": "C2"}.get(nm, "k"))
    if len(qp) >= 8:
        rho, p = stats.spearmanr(qp, qm)
        say(f"  quadrants of used events (each normalised by its event mean): "
            f"Spearman rho(penumbral growth, moat mature) = {rho:+.2f} (p = {p:.2f}, n = {len(qp)})")
    ax[2].set(xlabel="quadrant penumbral growth / event mean",
              ylabel="quadrant moat mature / event mean",
              title="quadrants (W blue, N orange, E red, S green)")
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rep_dir / "correlation.png", dpi=130)
    plt.close(fig)

    lead = {eid: leading_or_trailing(cfg, eid) for eid in ids}

    # ---- 3
    say("\n3. BIAS TEST: W-minus-E quadrant moat asymmetry in mature epochs (background flow removed)")
    say("   a residual disk-centre bias gives A > 0 east of the meridian and A < 0 west of it;")
    say("   a physical asymmetry keeps its sign")
    say("event      used   A east [km/s] (n)        A west [km/s] (n)        lon range")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    pool = {"east": [], "west": []}
    allA, allbg = [], []
    for eid in ids:
        f = event_dir(cfg, eid) / "sector_series.npz"
        if not f.exists():
            continue
        out = event_dir(cfg, eid)
        ser = dict(np.load(f))
        npz = np.load(out / "onset_series.npz")
        on = analyse(npz)
        t_mat = np.nanmax([on["t_pen_h"], on["t_moat_h"]]) + MATURE_AFTER_H
        if not np.isfinite(t_mat):
            continue
        lon = lon_series(events[eid], window_t0(out), ser["t_h"])
        r = ew_asymmetry(ser, lon, t_mat)
        if r is None:
            continue
        summ.setdefault(eid, {})["bias"] = {"east": r["east"], "west": r["west"]}
        e_, w_ = r["east"], r["west"]
        say(f"{eid:10s} {'yes' if used.get(eid) else 'no ':4s}  {e_[0]:+.3f} +- {e_[1]:.3f} ({e_[2]:3d})"
            f"     {w_[0]:+.3f} +- {w_[1]:.3f} ({w_[2]:3d})     {lon[0]:+.0f} .. {lon[-1]:+.0f} deg")
        sel = r["mature"] & np.isfinite(r["A"])
        ax[0].plot(lon[sel], r["A"][sel], ".", ms=3, label=eid)
        ax[1].plot(r["bg_vx"][sel] * 1000, r["A"][sel], ".", ms=3)
        if used.get(eid) or eid == "AR13010":
            for side in ("east", "west"):
                s = sel & ((lon < -MERIDIAN_GAP_DEG) if side == "east" else (lon > MERIDIAN_GAP_DEG))
                pool[side] += list(r["A"][s])
            allA += list(r["A"][sel])
            allbg += list(r["bg_vx"][sel])
    for side in ("east", "west"):
        v = np.array(pool[side])
        if len(v) > 2:
            say(f"  pooled {side:4s} of meridian: A = {v.mean():+.3f} +- {stats.sem(v):.3f} km/s "
                f"(n = {len(v)} epochs, {np.mean(v > 0) * 100:.0f} % positive)")
    if len(allA) > 10:
        allA, allbg = np.array(allA), np.array(allbg)
        g = np.isfinite(allA) & np.isfinite(allbg)
        lr = stats.linregress(allbg[g], allA[g])
        say(f"  regression A = c + beta * <v_x>_background: c = {lr.intercept:+.3f} km/s, "
            f"beta = {lr.slope:+.2f} +- {lr.stderr:.2f} (r = {lr.rvalue:+.2f}); a residual bias "
            "proportional to the measured one gives beta > 0")
    say("  NOTE: within an event, east of the meridian also means earlier, so the")
    say("  E/W split and the regression mix a residual bias with the moat's own")
    say("  development (the E-side moat catches up late). The N-S test below does not.")

    # hemisphere test: the disk-centre bias points towards the equator, so
    # a residual gives v_S - v_N > 0 for northern spots and < 0 for southern
    # ones, at any time; the spot's own development has no hemisphere sign
    say("\n   N-S test: S-minus-N quadrant moat asymmetry, mature epochs; a residual bias")
    say("   follows the hemisphere (N: > 0, S: < 0); the measured background <v_y> shows")
    say("   the bias itself (N: < 0, S: > 0)")
    say("event      used  hemi   v_S - v_N [km/s] (n)      background <v_y> [m/s]   follows hemisphere?")
    agree = []
    for eid in ids:
        f = event_dir(cfg, eid) / "sector_series.npz"
        if not f.exists():
            continue
        ser = dict(np.load(f))
        names = list(ser["names"])
        if "S" not in names or "N" not in names:
            continue
        on = analyse(np.load(event_dir(cfg, eid) / "onset_series.npz"))
        t_mat = np.nanmax([on["t_pen_h"], on["t_moat_h"]]) + MATURE_AFTER_H
        valid = ser["valid"].astype(bool)
        sel = valid & (ser["t_h"] >= t_mat) if np.isfinite(t_mat) else valid
        dns = ser["v"][:, names.index("S")] - ser["v"][:, names.index("N")]
        g = sel & np.isfinite(dns)
        if g.sum() < 3:
            continue
        lat = float(events[eid]["lat_ref"])
        m, se = float(np.mean(dns[g])), float(stats.sem(dns[g]))
        bgy = float(np.nanmedian(ser["bg_vy"][valid])) * 1000
        fol = (m > 0) == (lat >= 0) and abs(m) > 2 * se
        if used.get(eid) or eid == "AR13010":
            agree.append(fol)
        say(f"{eid:10s} {'yes' if used.get(eid) else 'no ':4s}  {'N' if lat >= 0 else 'S'}     "
            f"{m:+.3f} +- {se:.3f} ({g.sum():3d})        {bgy:+5.0f}                "
            f"{'yes' if fol else 'no'}")
    if agree:
        say(f"  {sum(agree)} of {len(agree)} events (used + AR13010) have a significant N-S "
            "asymmetry with the bias sign")
    say("  NOTE: Joy's law puts the following polarity poleward of a leading spot, so a")
    say("  moat weakened towards the following polarity ALSO gives this hemisphere sign.")

    # direction test: azimuth of the strongest mature moat (first harmonic
    # over the 30 deg sectors) vs (a) towards disk centre [residual bias]
    # and (b) away from the opposite-polarity flux [physical]
    say("\n   Direction test: azimuth of the strongest mature moat (first harmonic of the")
    say("   30 deg sectors) vs towards disk centre (residual bias) and away from the")
    say("   opposite-polarity flux (moat weak on the side facing it); 0 = W, 90 = N")
    say("event      used  phi_max  amp[km/s]  disk-centre  away-from-opp   |d| bias  |d| phys  mean lon")
    dbias, dphys = [], []

    def adiff(a, b):
        return abs((a - b + 180) % 360 - 180)
    for eid in ids:
        f = event_dir(cfg, eid) / "sector_series.npz"
        L = lead.get(eid, {})
        if not f.exists() or "phi_opp_deg" not in L:
            continue
        ser = dict(np.load(f))
        out = event_dir(cfg, eid)
        on = analyse(np.load(out / "onset_series.npz"))
        t_mat = np.nanmax([on["t_pen_h"], on["t_moat_h"]]) + MATURE_AFTER_H
        valid = ser["valid"].astype(bool)
        sel = valid & (ser["t_h"] >= t_mat) if np.isfinite(t_mat) else valid
        if sel.sum() < 3:
            continue
        with np.errstate(all="ignore"):
            prof = np.nanmean(ser["az_v"][sel], axis=0)
        phi = np.deg2rad(ser["az_deg"])
        g = np.isfinite(prof)
        a = 2 * np.mean(prof[g] * np.cos(phi[g]))
        b = 2 * np.mean(prof[g] * np.sin(phi[g]))
        phi_max = float(np.rad2deg(np.arctan2(b, a)) % 360)
        lon = lon_series(events[eid], window_t0(out), ser["t_h"])
        lon_m = float(np.mean(lon[sel]))
        lat = np.deg2rad(float(events[eid]["lat_ref"]))
        phi_dc = float(np.rad2deg(np.arctan2(-np.sin(lat),
                                              -np.cos(lat) * np.sin(np.deg2rad(lon_m)))) % 360)
        phi_away = (L["phi_opp_deg"] + 180) % 360
        db, dp = adiff(phi_max, phi_dc), adiff(phi_max, phi_away)
        if used.get(eid) or eid == "AR13010":
            dbias.append(db)
            dphys.append(dp)
        say(f"{eid:10s} {'yes' if used.get(eid) else 'no ':4s}  {phi_max:5.0f}    {np.hypot(a, b):.3f}"
            f"       {phi_dc:5.0f}         {phi_away:5.0f}          {db:4.0f}      {dp:4.0f}     {lon_m:+5.0f}")
    if dbias:
        say(f"  median angular distance of the strongest moat from: disk centre {np.median(dbias):.0f} deg,"
            f" away-from-opposite-polarity {np.median(dphys):.0f} deg (n = {len(dbias)}; random: 90 deg)")

    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvline(0, color="k", lw=0.5)
    ax[0].set(xlabel="longitude [deg]", ylabel="v$_W$ - v$_E$ [km/s]",
              title="E-W moat asymmetry, mature epochs")
    ax[0].legend(fontsize=6, ncol=2, markerscale=3)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xlabel="background <v$_x$> removed [m/s]", ylabel="v$_W$ - v$_E$ [km/s]",
              title="asymmetry vs the shrinking-Sun flow")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(rep_dir / "bias_test.png", dpi=130)
    plt.close(fig)

    # ---- 4
    say("\n4. LEADING OR TRAILING (magnetogram at the seed epoch)")
    say("event      used  B_umbra  pol  hemi  Hale leading  spot - opp. flux  same - opp. flux  flux opp/same  -> position / Hale")
    for eid in ids:
        r = lead[eid]
        summ.setdefault(eid, {})["lead"] = r
        if "note" in r:
            say(f"{eid:10s} {r['note']}")
            continue
        say(f"{eid:10s} {'yes' if used.get(eid) else 'no ':4s} {r['b_umbra_G']:+7.0f}  {'+' if r['polarity'] > 0 else '-'}"
            f"    {'N' if r['lat'] >= 0 else 'S'}       {'+' if r['hale_leading_polarity'] > 0 else '-'}"
            f"         {r['dx_spot_minus_opp_Mm']:+6.1f} Mm         {r['dx_same_minus_opp_Mm']:+6.1f} Mm"
            f"        {r['flux_ratio_opp_over_same']:5.2f}      -> {r['by_position']} / {r['by_hale']}")

    (rep_dir / "report.txt").write_text(buf.getvalue())
    print(f"\n-> {rep_dir / 'report.txt'}, correlation.png, bias_test.png")


if __name__ == "__main__":
    main()
