# moat-flow-onset

Pipeline to download and track SDO/HMI observations of pores that develop
penumbrae, in order to measure how the onset of the **moat flow** relates to
the onset of **penumbra formation** in the photosphere.

Science question: for a sample of ~20–30 pores that turn into full sunspots,
does the organized radial outflow (moat flow, traced by moving magnetic
features and granulation flows) appear before, together with, or after the
first stable penumbral sector?

## Data strategy

Two acquisition paths per event, both driven by the same event catalog:

| Purpose | Series | Cadence | How |
|---|---|---|---|
| Flow tracking (LCT), MMF tracking | `hmi.Ic_45s`, `hmi.M_45s` | 45 s | JSOC export with server-side `im_patch` tracked cutouts |
| Vector field context, spot masks, metadata | `hmi.sharp_cea_720s` | 720 s | Direct SHARP download (`Br`, `continuum`, `magnetogram`, `bitmap`) |

Rationale: granulation decorrelates on ~5–10 min timescales, so correlation
tracking of continuum images needs the 45 s series — the 12-min SHARP cadence
is too slow for granular LCT. The SHARP series provides the tracked CEA
geometry, disambiguated `Br`, and the AR bounding box used to define the
`im_patch` cutout.

Event selection constraint: the pore→penumbra transition must happen at
small heliocentric angle (default cut |Stonyhurst longitude| ≤ 40°) so that
LCT foreshortening stays manageable.

Data volume: ~2–3 days per event at 45 s cadence, two series, ~512×512 patch
≈ 10–15 GB/event, i.e. a few hundred GB for the full sample. Development is
local on 1–2 events; the full sample runs on the cluster (all paths are set
in `config.yaml`, nothing is hardcoded).

## Tracking

- **FLCT** via [`pyflct`](https://github.com/PyDL/pyflct) (Fisher & Welsch
  2008) on continuum (granulation) and on |Blos| (magnetic features):
  pairwise flows at 45–90 s separation, averaged over 1–2 h windows,
  Gaussian window σ ≈ 1–2 Mm.
- **DeepVel** as a second, independent velocity estimate
  (`moatflow.tracking.deepvel` defines the interface; pretrained models are
  cadence-specific, so it is wired for the 45 s continuum cubes).

## Analysis

- `moatflow.analysis.penumbra`: continuum intensity segmentation
  (umbra/penumbra thresholds relative to quiet sun) → penumbral area vs time
  → penumbra onset time.
- `moatflow.analysis.moat`: azimuthally averaged radial velocity profiles
  around the spot barycenter from the FLCT flow maps → mean outflow speed in
  an annulus outside the spot boundary vs time → moat onset time.

## Event catalog

`catalog/events.yaml` is the single source of truth. Each entry needs only a
NOAA AR number and (optionally) a manual time window;
`scripts/resolve_event.py` fills in the HARP number (from the official JSOC
NOAA↔HARP mapping) and the disk-passage time range, and applies the
longitude cut.

Seed events come from the penumbra-formation literature:

- NOAA 11490 (May 2012) — penumbra formation and Evershed-flow onset,
  Murabito et al. 2016, ApJ 825, 75 ([arXiv:1604.05610](https://arxiv.org/abs/1604.05610)).
- Murabito et al. 2018, ApJ 855, 58 — sample of 12 β-type ARs from 2011–2012
  with observed penumbra formation. **TODO: transcribe Table 1 of that paper
  into `events.yaml`** (not machine-readable from the web).
- Review: Murabito et al. 2019 ([arXiv:1901.05207](https://arxiv.org/abs/1901.05207));
  onset study (pre-HMI benchmark AR 11024): García-Rivas et al. 2024
  ([arXiv:2403.18455](https://arxiv.org/abs/2403.18455)).

Beyond the literature seed, extend the sample by querying SHARP metadata for
emerging ARs (flux growth + longitude window) and vetting quicklook movies.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml   # then edit: jsoc_email, data_root
```

`pyflct` needs the FLCT C library; `pip install pyflct` ships wheels for
common platforms (macOS arm64 included).

## Workflow

```bash
# 1. Fill in HARPNUM + time window for catalog entries
python scripts/resolve_event.py AR11490

# 2. Download SHARPs (720 s) and 45 s tracked cutouts for one event
python scripts/download_event.py AR11490 --sharps --patches

# 3. Build HDF5 cubes and run FLCT
python scripts/run_flct.py AR11490 --segment continuum
```
