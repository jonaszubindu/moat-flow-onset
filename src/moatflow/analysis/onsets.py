"""Onset times of the penumbral-area and moat-outflow curves, and their lag.

The lag between the two onsets is the answer to the science question, so
both onsets go through IDENTICAL machinery. Pairing a departure-from-noise
criterion on one curve with a fraction-of-mature criterion on the other
would bias the lag by construction. For each curve (hourly epochs):

  1. smooth   3-epoch NaN-aware running mean; epochs where tracking was
              lost stay NaN, so they can never count as "above".
  2. zero Z   the state with no penumbra / no moat:
                penumbral area -- median of the first BASELINE_H valid
                  hours. In the pore phase the 0.9 I_qs threshold already
                  counts the grey rim of pores as "penumbra" (10-30 Mm^2),
                  so this level is an artefact to subtract.
                moat outflow  -- 0 km/s, i.e. no flow. The slightly
                  negative pre-onset values are a real converging inflow,
                  not an offset; subtracting them would make the onset
                  fire when the inflow ends rather than when outflow starts.
  3. mature M 90th percentile of the smoothed valid curve. Robust to late
              decay or tracking loss at the end of the window, which the
              mean of the last hours is not (AR11210: area peaks at
              138 Mm^2 but its last-20-h mean is 55).
  4. threshold  Z + f (M - Z), default f = 0.4
  5. onset    first epoch starting PERSIST consecutive valid epochs above
              the threshold.

The lag is reported for several f as a robustness check: if its sign holds
across f, the conclusion does not depend on where the threshold is put.

Second view, "first appearance", for both curves alike: penumbra =
baseline + max(3 sigma, 20 Mm^2); moat = 3 sigma of the smoothed curve's
noise above zero, at least 0.05 km/s. The 40 % rule times when each
quantity is ESTABLISHED, which also depends on how fast it grows (a large
spot accumulates its penumbral area slowly); first appearance times when
it is first DETECTABLE. Both lags are reported.

Each onset must be preceded by BASELINE_H tracked hours, for both curves:
an onset earlier than that cannot be told apart from "already on when
tracking started".
"""

import numpy as np

SMOOTH_N = 3
PERSIST = 3
BASELINE_H = 8.0
MATURE_PCT = 90
FRACTION = 0.4
FRACTION_SWEEP = (0.25, 0.4, 0.6)
FIRST_SIGMA = 3.0
FIRST_MIN_AREA = 20.0          # Mm^2
MOAT_FIRST_FLOOR = 0.05        # km/s, ~3x the hour-to-hour scatter
STABLE_H = 12.0                # an onset should hold this long
CONTAM_AT_ONSET = 0.3          # annulus fraction masked near the onset

# quality thresholds for flags
NO_PORE_PHASE_FRAC = 0.3       # baseline area / mature area above this
WEAK_RING_SNR = 1.0            # mature moat / sector scatter below this
SHORT_BASELINE_H = 12.0        # first valid epoch to literature start
HIGH_CONTAMINATION = 0.3       # max fraction of the annulus masked
LOW_DOPPLER_R = 0.5


def smooth(y, n=SMOOTH_N):
    """NaN-aware running mean; NaN wherever the input is NaN."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    h = n // 2
    for i in np.flatnonzero(np.isfinite(y)):
        out[i] = np.nanmean(y[max(0, i - h):i + h + 1])
    return out


def first_persistent(t, ys, thr, persist=PERSIST):
    """Time of the first epoch that starts `persist` epochs above `thr`."""
    above = np.isfinite(ys) & (ys > thr)
    for i in range(len(above) - persist + 1):
        if above[i:i + persist].all():
            return float(t[i])
    return np.nan


def pore_baseline(t, y, hours=BASELINE_H):
    """(median, robust sigma, first valid time) over the first `hours`."""
    good = np.isfinite(y)
    if not good.any():
        return np.nan, np.nan, np.nan
    t0 = float(t[good][0])
    vals = y[good & (t < t0 + hours)]
    med = float(np.median(vals))
    sig = float(1.4826 * np.median(np.abs(vals - med)))
    return med, sig, t0


def curve_onset(t, y, zero, fraction=FRACTION):
    """Onset of one curve with the fraction-of-mature rule (see module doc)."""
    ys = smooth(y)
    good = np.isfinite(ys)
    if good.sum() < PERSIST:
        return {"t": np.nan, "thr": np.nan, "mature": np.nan,
                "zero": zero, "smooth": ys}
    mature = float(np.nanpercentile(ys[good], MATURE_PCT))
    thr = zero + fraction * (mature - zero)
    return {"t": first_persistent(t, ys, thr), "thr": thr, "mature": mature,
            "zero": zero, "smooth": ys}


def moat_first_appearance(t, v, valid):
    """First appearance of outflow: 3 sigma of the smoothed curve's noise
    above zero (at least MOAT_FIRST_FLOOR).

    sigma is estimated from the residual about the 3-point running mean
    (white noise leaves 0.816 sigma there) and divided by sqrt(3) for the
    smoothed curve the threshold is applied to.
    """
    v = np.where(valid, v, np.nan)
    ys = smooth(v)
    res = v - ys
    good = np.isfinite(res)
    if good.sum() < 10:
        return np.nan, np.nan
    r = res[good]
    sig_raw = 1.4826 * np.median(np.abs(r - np.median(r))) / 0.816
    sig_smoothed = sig_raw / np.sqrt(3.0)
    thr = max(3.0 * sig_smoothed, MOAT_FIRST_FLOOR)
    return first_persistent(t, ys, thr), thr


def holds_above(t, ys, thr, t_on, hours=STABLE_H):
    """False if the smoothed curve drops back below thr within `hours`."""
    if not np.isfinite(t_on):
        return np.nan
    sel = (t > t_on) & (t <= t_on + hours) & np.isfinite(ys)
    return bool(np.all(ys[sel] > thr)) if sel.any() else np.nan


def penumbra_onset(t, a_pen, valid, fraction=FRACTION):
    a = np.where(valid, a_pen, np.nan)
    base, sig, t_first_valid = pore_baseline(t, a)
    res = curve_onset(t, a, zero=base, fraction=fraction)
    res["baseline_sigma"] = sig
    res["t_first_valid"] = t_first_valid
    res["t_first_departure"] = first_persistent(
        t, res["smooth"], base + max(FIRST_SIGMA * sig, FIRST_MIN_AREA))
    res["baseline_frac"] = (base / res["mature"]
                            if np.isfinite(res["mature"]) and res["mature"] > 0
                            else np.nan)
    return res


def moat_onset(t, v, valid, fraction=FRACTION):
    return curve_onset(t, np.where(valid, v, np.nan), zero=0.0,
                       fraction=fraction)


def analyse(npz, fraction=FRACTION, sweep=FRACTION_SWEEP, verify=None):
    """All onset numbers, quality figures and flags for one event.

    npz    : loaded onset_series.npz (dict-like)
    verify : parsed verify_flct.json, or None
    Returns a flat dict; 'blocking' flags mean the lag is not measurable.
    """
    t = np.asarray(npz["t_h"], dtype=float)
    valid = np.asarray(npz["valid"], dtype=bool)
    pen = penumbra_onset(t, npz["a_pen"], valid, fraction)
    moat = moat_onset(t, npz["v_gran"], valid, fraction)
    has_mmf = "v_mmf" in npz and np.isfinite(npz["v_mmf"]).any()
    mmf = moat_onset(t, npz["v_mmf"], valid, fraction) if has_mmf else None
    pf0, pf1 = (float(x) for x in npz["pf_lit"]) if "pf_lit" in npz \
        else (np.nan, np.nan)

    t_moat_first, moat_first_thr = moat_first_appearance(t, npz["v_gran"],
                                                         valid)
    r = {
        "n_epochs": int(len(t)),
        "n_valid": int(valid.sum()),
        "t_first_valid_h": pen["t_first_valid"],
        "t_last_valid_h": float(t[valid][-1]) if valid.any() else np.nan,
        "lit_start_h": pf0, "lit_end_h": pf1,
        "pen_baseline_Mm2": pen["zero"],
        "pen_mature_Mm2": pen["mature"],
        "pen_baseline_frac": pen["baseline_frac"],
        "t_pen_h": pen["t"],
        "t_pen_first_departure_h": pen["t_first_departure"],
        "moat_mature_km_s": moat["mature"],
        "t_moat_h": moat["t"],
        "t_mmf_h": mmf["t"] if mmf else np.nan,
        "lag_moat_h": moat["t"] - pen["t"],
        "lag_mmf_h": (mmf["t"] - pen["t"]) if mmf else np.nan,
        "t_moat_first_h": t_moat_first,
        "moat_first_thr_km_s": moat_first_thr,
        "lag_first_h": t_moat_first - pen["t_first_departure"],
        "pre_onset_pen_h": pen["t"] - pen["t_first_valid"],
        "pre_onset_moat_h": moat["t"] - pen["t_first_valid"],
        "pen_holds": holds_above(t, pen["smooth"], pen["thr"], pen["t"]),
        "moat_holds": holds_above(t, moat["smooth"], moat["thr"], moat["t"]),
    }
    for f in sweep:
        p = penumbra_onset(t, npz["a_pen"], valid, f)["t"]
        m = moat_onset(t, npz["v_gran"], valid, f)["t"]
        r[f"lag_f{int(round(f * 100)):02d}_h"] = m - p

    # quality of the moat measurement
    fc = np.asarray(npz["frac_contaminated"]) if "frac_contaminated" in npz \
        else np.array([np.nan])
    sc = np.asarray(npz["sector_scatter"]) if "sector_scatter" in npz \
        else np.array([np.nan])
    moat_on = np.isfinite(moat["smooth"]) & (moat["smooth"] > moat["thr"])
    scatter = float(np.nanmedian(sc[moat_on])) if moat_on.any() else np.nan
    if np.isfinite(moat["t"]) and fc.shape == t.shape:
        near = np.abs(t - moat["t"]) <= 3
        r["contamination_at_moat_onset"] = (float(np.nanmax(fc[near]))
                                            if near.any() else np.nan)
    else:
        r["contamination_at_moat_onset"] = np.nan
    r["contamination_median"] = float(np.nanmedian(fc))
    r["contamination_max"] = float(np.nanmax(fc))
    r["ring_snr"] = (moat["mature"] / scatter
                     if np.isfinite(scatter) and scatter > 0 else np.nan)
    if verify:
        dp, ss = verify.get("doppler"), verify.get("shrinking_sun")
        r["doppler_r"] = dp["r"] if dp else np.nan
        r["doppler_amp"] = dp["flct_over_doppler"] if dp else np.nan
        r["shrink_sun_shift_m_s"] = ss["max_curve_shift_m_s"] if ss else np.nan
    else:
        r["doppler_r"] = r["doppler_amp"] = r["shrink_sun_shift_m_s"] = np.nan

    blocking, warnings = [], []
    if not np.isfinite(pen["t"]):
        blocking.append("no penumbra onset")
    if not np.isfinite(moat["t"]):
        blocking.append("no moat onset")
    if np.isfinite(pen["baseline_frac"]) and \
            pen["baseline_frac"] > NO_PORE_PHASE_FRAC:
        blocking.append(f"no pore phase (baseline {pen['baseline_frac']:.0%} "
                        "of mature area)")
    if np.isfinite(pen["t"]) and pen["t"] < pen["t_first_valid"] + BASELINE_H:
        blocking.append("penumbra onset inside the baseline window")
    # same requirement as for the penumbra: an onset needs BASELINE_H
    # tracked hours before it, else "already on" cannot be excluded
    if np.isfinite(moat["t"]) and \
            moat["t"] < pen["t_first_valid"] + BASELINE_H:
        blocking.append(f"moat onset only {r['pre_onset_moat_h']:.0f} h "
                        "after tracking starts (may already be on)")
    if np.isfinite(pf0):
        lead = pf0 - pen["t_first_valid"]
        if lead < 0:
            warnings.append(f"tracking starts {-lead:.0f} h after the "
                            "literature formation start")
        elif lead < SHORT_BASELINE_H:
            warnings.append(f"only {lead:.0f} h tracked before the "
                            "literature formation start")
    if np.isfinite(r["ring_snr"]) and r["ring_snr"] < WEAK_RING_SNR:
        warnings.append(f"weak moat (ring SNR {r['ring_snr']:.1f})")
    if np.isfinite(r["contamination_max"]) and \
            r["contamination_max"] > HIGH_CONTAMINATION:
        warnings.append(f"annulus up to {r['contamination_max']:.0%} "
                        "contaminated")
    if np.isfinite(r["doppler_r"]) and r["doppler_r"] < LOW_DOPPLER_R:
        warnings.append(f"Doppler check r = {r['doppler_r']:.2f}")
    if r["moat_holds"] is False:
        warnings.append(f"moat drops back below threshold within "
                        f"{STABLE_H:.0f} h of its onset")
    if r["pen_holds"] is False:
        warnings.append(f"penumbra drops back below threshold within "
                        f"{STABLE_H:.0f} h of its onset")
    if np.isfinite(r["contamination_at_moat_onset"]) and \
            r["contamination_at_moat_onset"] >= CONTAM_AT_ONSET:
        warnings.append(f"annulus {r['contamination_at_moat_onset']:.0%} "
                        "contaminated within 3 h of the moat onset")
    if np.isfinite(r["lag_moat_h"]) and np.isfinite(r["lag_first_h"]) and \
            abs(r["lag_moat_h"]) > 2 and abs(r["lag_first_h"]) > 2 and \
            np.sign(r["lag_moat_h"]) != np.sign(r["lag_first_h"]):
        warnings.append("lag sign differs between 40 % rule and first "
                        "appearance")
    sweep_lags = [r[f"lag_f{int(round(f * 100)):02d}_h"] for f in sweep]
    signs = {np.sign(x) for x in sweep_lags if np.isfinite(x) and abs(x) > 2}
    if len(signs) > 1:
        warnings.append("lag sign depends on the threshold fraction")
    r["blocking"] = blocking
    r["warnings"] = warnings
    r["_curves"] = {"t": t, "pen": pen, "moat": moat, "mmf": mmf}
    return r
