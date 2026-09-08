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
  with observed penumbra formation; Table 1 is in `events.yaml`, extracted
  from the IOP full text. **Verify against the PDF before production runs**
  — AR 11243's row came out wrong (see `VETTING.md` section A).
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

## Cluster deployment

Only **one line** changes: `data_root` in `config.yaml`. No path is
hardcoded anywhere in `src/` or `scripts/` — everything else derives from
`REPO_ROOT`, which the package computes from its own module location, so
it follows wherever the repo is cloned. Absolute `data_root` values pass
through as given; relative ones resolve against the repo root.

```bash
git clone git@github.com:jonaszubindu/moat-flow-onset.git
cd moat-flow-onset
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml     # config.yaml is gitignored — this IS the setup step
# then edit one line:  data_root: /sml/zbindenj/moatflow
```

`catalog/events.yaml` travels with the clone already resolved (HARP
numbers, windows, patch anchors), so `resolve_event.py` need not be
re-run. `jsoc_email` stays the same.

The project root (holding `catalog/events.yaml` and `config.yaml`) is
found from `$MOATFLOW_ROOT` if set, else by walking up from the working
directory, else from the package location. So a plain `pip install .`
works as long as you run inside the checkout; for batch jobs that start
elsewhere, set it explicitly:

```bash
export MOATFLOW_ROOT=/sml/zbindenj/moat-flow-onset
```

Four environment gotchas that are not about paths:

1. **`pyflct`** needs the FLCT C library. Verify before queuing a long
   job: `python -c "import pyflct"`.
2. **`ffmpeg`** is required only for the movies (`quicklook.py`,
   `moat_movie.py`) and is often absent on clusters — `module load
   ffmpeg` or `conda install ffmpeg`. Cubes, FLCT, onset analysis and
   PNG figures work without it; matplotlib is pinned to the headless
   `Agg` backend.
3. **JSOC serializes exports per registered e-mail** — one pending
   export at a time. Parallel array jobs downloading different events
   will collide; run downloads as one sequential job, or stagger them.
4. **Volume**: ~10-15 GB of FITS per event per 45 s series, plus a cube
   of the same size. `cadence: 90s` in `config.yaml` halves it. Raw
   FITS may be deleted once a cube is built and vetted (the cube carries
   data, times and the reference WCS) — keep them only if per-frame WCS
   is needed, e.g. for the Doppler deprojection.

## Workflow

Per event, in order:

```bash
# 1. Resolve HARPNUM + download window (only for new catalog entries)
python scripts/resolve_event.py AR11490

# 2. Download. SHARPs are cheap (vetting/context); the 45 s series carry
#    the science — hmi.Ic_45s for granulation LCT, hmi.M_45s for MMFs.
python scripts/download_event.py AR11490 --sharps
python scripts/download_event.py AR11490 --patches --series hmi.Ic_45s

# 3. Vet: builds the QUALITY-filtered cube, reports cadence gaps and
#    scene drift, writes movie + summary figures.
python scripts/quicklook.py AR11490 --series hmi.Ic_45s

# 4. Flow maps (hourly windows, sigma 5 px). Add --thresh 30 for
#    magnetograms so sparse MMFs are not diluted by noise pixels.
python scripts/run_flct.py AR11490 --series hmi.Ic_45s --window 3600 --sigma 5

# 5. The science product: tracked penumbral area vs moat outflow
#    (both tracers), -> onset_series.npz + onset_comparison.png
python scripts/onset_analysis.py AR11490

# 6. MANDATORY verification of that curve: shows frame by frame which
#    pixels enter the annulus average, the per-sector breakdown, and the
#    radial profile -> moat_audit.mp4 + moat_audit.npz
python scripts/moat_audit.py AR11490
```

Step 6 is not optional. An azimuthal mean over an annulus can look like a
moat while being produced by something else — a couple of hot sectors, a
neighbouring pore inside the annulus, or an annulus sitting beside the
actual flow peak rather than on it. The audit movie shows all three at
once, and its `.npz` records the contaminated fraction and the difference
between the masked and unmasked curve, so the contamination is a number
in the paper rather than an assumption. What a healthy moat looks like:
**all sectors positive and comparable** (a full ring in the polar plot),
the **radial profile peaking inside the shaded annulus**, and **arrows
radiating outward** on the image.

Optional, per event or once:

```bash
python scripts/vet_batch.py                    # SHARP + quicklook for every event
python scripts/verify_flct.py AR11490 --lon0 -35 --lat -13.1   # Doppler + divergence checks
python scripts/moat_movie.py AR11490           # slow explanatory tracking movie
```

`scripts/example_m45s_sharp_fov.py` is a standalone (moatflow-free)
example of the JSOC `im_patch` recipe, useful for handing the download
method to someone else.
