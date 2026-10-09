"""Load one event's cube, flows and track the way onset_analysis does.

Scripts that re-measure an event (sector_analysis, ...) must follow the
same spot on the same epochs as onset_analysis, or their numbers are not
comparable: same flow epochs, seed (seed_epoch, stored in
onset_series.npz), seed rule (catalog seed_prefer_west) and magnetogram
flows paired with the continuum epochs by time.
"""

from datetime import datetime

import h5py
import numpy as np

from ..batch import CUBE_IC, FLOW_IC
from ..catalog import load_events
from ..config import event_dir
from ..cubes import load_cube
from ..viz import parse_datetimes, parse_times
from .spottrack import seed_epoch, track_spot

FLOW_M = "flct_hmi.M_45s_magnetogram_w3600s_s5px_k1.h5"


def literature_hours(event, t0):
    """(start, end) of the literature formation interval in hours since
    t0; NaN where the catalog has no value."""
    def h(key):
        if not event.get(key):
            return np.nan
        return (datetime.fromisoformat(str(event[key])) - t0
                ).total_seconds() / 3600
    return h("t_penumbra_start"), h("t_penumbra_end")


def aligned_flows(flow_file, t_mid, t0, tol_s=900.0):
    """(VX, VY, n_missing): a flow file's maps matched to the epochs
    t_mid [s since t0] by absolute time, NaN maps where none lies within
    tol_s. Each flow file's t_mid_s counts from its own cube's first
    frame (attr t0_isot); cubes of one event can differ in frame set."""
    with h5py.File(flow_file) as f:
        tm = f["t_mid_s"][:]
        if "t0_isot" in f.attrs:
            tm = tm + (datetime.fromisoformat(str(f.attrs["t0_isot"]))
                       - t0).total_seconds()
        vx, vy = f["vx"][:], f["vy"][:]
    j = np.array([int(np.argmin(np.abs(tm - t))) for t in t_mid])
    ok = np.abs(tm[j] - np.asarray(t_mid)) <= tol_s
    VX = np.where(ok[:, None, None], vx[j], np.nan)
    VY = np.where(ok[:, None, None], vy[j], np.nan)
    return VX, VY, int((~ok).sum())


def load_tracked_event(cfg, event_id, seed_h=None, with_mmf=True):
    """Everything an analysis needs for one event, tracked."""
    out = event_dir(cfg, event_id)
    event = load_events()[event_id]
    cube, t_iso, _ = load_cube(out / CUBE_IC)
    times_s = parse_times(t_iso)
    t0 = parse_datetimes(t_iso[:1])[0]
    with h5py.File(out / FLOW_IC) as f:
        t_mid, VX, VY = f["t_mid_s"][:], f["vx"][:], f["vy"][:]
    t_h = t_mid / 3600
    pf = literature_hours(event, t0)
    idx = [int(np.argmin(np.abs(times_s - t))) for t in t_mid]
    frames = [np.asarray(cube[i]) for i in idx]
    seed = seed_epoch(t_h, pf[1], override=seed_h,
                      npz_path=out / "onset_series.npz")
    tr = track_spot(frames, int(np.argmin(np.abs(t_h - seed))),
                    prefer_west=event.get("seed_prefer_west", True))
    VXm = VYm = None
    if with_mmf and (out / FLOW_M).exists():
        VXm, VYm, n_miss = aligned_flows(out / FLOW_M, t_mid, t0)
        if n_miss:
            print(f"MMF: {n_miss} of {len(t_mid)} epochs have no "
                  "magnetogram flow map within 15 min (left empty)")
    return {"out": out, "event": event, "frames": frames, "t_iso": t_iso,
            "epoch_idx": idx, "t0": t0, "t_mid": t_mid, "t_h": t_h,
            "VX": VX, "VY": VY, "VXm": VXm, "VYm": VYm, "tr": tr,
            "pf_lit": pf, "seed_h": float(t_h[tr["seed_idx"]])}
