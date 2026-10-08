"""Assemble downloaded FITS series into one HDF5 cube per event/series.

Layout (one file, e.g. AR11490/cube_hmi.Ic_45s_continuum.h5):
    data      (nt, ny, nx)  float32
    t_obs     (nt,)         ISO-8601 strings (TAI)
    attrs     reference FITS header of the first frame (patch WCS)

Half a million individual FITS files do not survive contact with a
cluster filesystem; the cubes are what tracking and analysis read.
"""

import glob
import os
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def _first_image_hdu(hdul):
    return next(h for h in hdul if h.data is not None)


def _trec_digits(s: str) -> str:
    """Normalize any T_REC representation to its digit string."""
    return "".join(ch for ch in str(s) if ch.isdigit())


def jsoc_bad_trecs(series: str, files: list, email: str) -> set[str]:
    """T_RECs (digit-normalized) with QUALITY != 0 over the files' span.

    The 45 s im_patch exports select records by TIME only — unlike our
    SHARP queries there is no QUALITY filter, so eclipse-season or
    calibration frames arrive as normal-looking FITS. Query the series
    keywords over the downloaded span and return the bad set so cube
    building can exclude them.
    """
    import re

    import drms

    trecs = sorted(_trec_digits(re.search(r"(\d{8}_\d{6})_TAI",
                                          os.path.basename(f)).group(1))
                   for f in files)
    t0, t1 = trecs[0], trecs[-1]
    fmt = lambda d: (f"{d[0:4]}.{d[4:6]}.{d[6:8]}_"
                     f"{d[8:10]}:{d[10:12]}:{d[12:14]}_TAI")
    # convert_numeric=False: drms otherwise turns the hex QUALITY strings
    # into NaN. Only the severe top-nibble flags (0xF0000000: missing /
    # unusable image, eclipse, calibration) disqualify a record — many
    # good 45 s records carry benign informational bits like 0x00010000.
    k = drms.Client(email=email).query(
        f"{series}[{fmt(t0)}-{fmt(t1)}@45s]", key="T_REC, QUALITY",
        convert_numeric=False)

    def severe(q: str) -> bool:
        q = str(q).strip()
        try:
            return bool(int(q, 16) & 0xF0000000)
        except ValueError:
            return True     # MISSING / unparsable -> treat as bad
    return {_trec_digits(t) for t, q in zip(k.T_REC, k.QUALITY) if severe(q)}


def build_cube(fits_dir: Path, out_file: Path,
               pattern: str = "*.fits",
               exclude_trecs: set[str] | None = None,
               deproject: bool = False, continuum: bool = False) -> Path:
    """deproject: remap every frame onto the solar surface while reading
    (moatflow.deproject; continuum frames are also limb-normalised), and
    record the per-frame centre/mu/limb fit as datasets."""
    import re

    files = sorted(glob.glob(str(fits_dir / pattern)))
    if exclude_trecs:
        n0 = len(files)
        files = [f for f in files
                 if _trec_digits(re.search(r"(\d{8}_\d{6})_TAI",
                                           os.path.basename(f)).group(1))
                 not in exclude_trecs]
        print(f"QC: excluded {n0 - len(files)} QUALITY!=0 frames")
    if not files:
        raise FileNotFoundError(f"No FITS files in {fits_dir}")

    with fits.open(files[0]) as hdul:
        hdu = _first_image_hdu(hdul)
        shape, header = hdu.data.shape, hdu.header
    if deproject:
        from .deproject import N_TARGET, PX_MM, deproject_frame
        out_shape = (N_TARGET, N_TARGET)
        info_keys = ("lon_c_deg", "lat_c_deg", "mu_c", "ld_a", "ld_b",
                     "n_filled")
        print(f"deprojecting {len(files)} frames onto a {N_TARGET}^2 "
              f"surface grid at {PX_MM} Mm/px"
              + (" (limb-normalised)" if continuum else ""))
    else:
        out_shape = shape

    # Build into a .partial file and rename only on success: a crash
    # (truncated FITS, full disk, Ctrl-C) otherwise leaves a half-filled
    # cube behind that looks valid to everything downstream, and its
    # unwritten frames carry empty timestamps.
    out_file = Path(out_file)
    tmp = out_file.with_name(out_file.name + ".partial")
    try:
      with h5py.File(tmp, "w") as h5:
        data = h5.create_dataset("data", shape=(len(files), *out_shape),
                                 dtype="f4", chunks=(1, *out_shape))
        t_obs = h5.create_dataset("t_obs", shape=(len(files),),
                                  dtype=h5py.string_dtype())
        if deproject:
            info = {k: h5.create_dataset(f"dp/{k}", shape=(len(files),),
                                         dtype="f8") for k in info_keys}
        for i, f in enumerate(files):
            # surface the offending file — truncated FITS from interrupted
            # downloads otherwise fail deep inside astropy
            try:
                with fits.open(f) as hdul:
                    hdu = _first_image_hdu(hdul)
                    if hdu.data.shape != shape:
                        raise ValueError(
                            f"shape {hdu.data.shape} != {shape} — "
                            "frame not tracked/cropped consistently?")
                    if deproject:
                        data[i], inf = deproject_frame(hdu.data, hdu.header,
                                                       continuum)
                        for k in info_keys:
                            info[k][i] = inf[k]
                    else:
                        data[i] = hdu.data.astype("f4")
                    t = hdu.header.get("T_OBS") or hdu.header.get("DATE-OBS")
                    if not t or not str(t).strip():
                        raise ValueError("header has neither T_OBS nor "
                                         "DATE-OBS")
                    t_obs[i] = t
            except Exception as e:
                raise RuntimeError(
                    f"unreadable FITS {f}: {e}\n"
                    "  Probably a truncated download. Repair with:\n"
                    "    python scripts/verify_downloads.py <EVENT> --fix\n"
                    "  then re-run the download command to refetch that "
                    "chunk.") from e
        for k, v in header.items():
            if k and not isinstance(v, fits.header._HeaderCommentaryCards):
                try:
                    h5.attrs[k] = v
                except TypeError:
                    pass
        if deproject:
            # the FITS keys above describe frame 0 on the CCD, not this grid
            h5.attrs.update({"DEPROJECTED": 1, "DP_PROJ": "ARC (Postel), "
                             "centred per frame on the im_patch centre, "
                             "north up", "DP_PX_MM": PX_MM,
                             "DP_LIMBNORM": int(continuum)})
      tmp.replace(out_file)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"Wrote {out_file}: {len(files)} frames of {out_shape}"
          + (" (deprojected)" if deproject else ""))
    return out_file


def cube_ok(path) -> bool:
    """True if a cube is complete: every frame has a timestamp.

    Cubes written before the build became atomic can be half-filled, and
    a half-filled one is indistinguishable from a good one by name alone.
    """
    try:
        with h5py.File(path, "r") as h5:
            if "data" not in h5 or "t_obs" not in h5:
                return False
            t = h5["t_obs"][:]
            if len(t) == 0 or len(t) != h5["data"].shape[0]:
                return False
            dec = lambda x: x.decode() if isinstance(x, bytes) else x
            return all(dec(x).strip() for x in t)
    except Exception:
        return False


def build_event_cube(event_dir: Path, series: str, segment: str,
                     email: str | None = None, deproject: bool = False,
                     raw_dir: Path | None = None) -> Path:
    """Build (or reuse) the cube for one event/series, QUALITY-filtered.

    For the 45 s series the im_patch export has no QUALITY filter, so
    bad records are fetched from JSOC and excluded here; SHARP cubes are
    already filtered at query time.

    deproject : remap onto the solar surface (catalog `deproject: true`;
                45 s series only -- SHARP CEA maps already are). The cube
                keeps its usual name so every later step reads it as is.
    raw_dir   : event directory holding the FITS (catalog `raw_from`),
                when it is not event_dir itself.
    """
    cube_file = event_dir / f"cube_{series}_{segment}.h5"
    deproject = deproject and series.startswith("hmi.")
    src_dir = (raw_dir or event_dir) / series
    if cube_file.exists():
        if cube_ok(cube_file):
            with h5py.File(cube_file, "r") as h5:
                is_dp = bool(h5.attrs.get("DEPROJECTED", 0))
            if is_dp != deproject:
                raise SystemExit(
                    f"{cube_file.name} is {'' if is_dp else 'not '}"
                    f"deprojected but the catalog says deproject: "
                    f"{str(deproject).lower()} -- delete the cube (and the "
                    "flct_*.h5 built from it) to rebuild")
            return cube_file
        print(f"existing {cube_file.name} is incomplete (interrupted build "
              "by an older version?) — rebuilding")
        cube_file.unlink()
    pattern = f"*.{segment}.fits"
    exclude = None
    if series.startswith("hmi.") and email:
        files = sorted(glob.glob(str(src_dir / pattern)))
        if files:
            exclude = jsoc_bad_trecs(series, files, email)
    return build_cube(src_dir, cube_file, pattern=pattern,
                      exclude_trecs=exclude, deproject=deproject,
                      continuum=segment == "continuum")


def load_cube(path: Path):
    """Return (data dataset [lazy], times as ISO strings, attrs dict)."""
    h5 = h5py.File(path, "r")
    return h5["data"], [t.decode() if isinstance(t, bytes) else t
                        for t in h5["t_obs"][:]], dict(h5.attrs)


def event_cube_opts(cfg: dict, event_id: str) -> dict:
    """build_event_cube options from the catalog entry: `deproject: true`
    and `raw_from: <event>` (read another event's FITS, e.g. to rebuild
    an event deprojected for validation without downloading it twice)."""
    from .catalog import load_events
    from .config import event_dir
    ev = load_events().get(event_id, {})
    raw = ev.get("raw_from")
    return {"deproject": bool(ev.get("deproject", False)),
            "raw_dir": event_dir(cfg, raw) if raw else None}
