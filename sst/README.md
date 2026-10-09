# SST/CRISP Fe I 6302: Milne-Eddington inversions

Scan-by-scan ME inversions of SSTRED narrowband cubes with
[pyMilne](https://github.com/jaimedelacruz/pyMilne) (de la Cruz Rodríguez
2019, A&A 631, A153), plus overview plots and movies. Meant for AR13010
and 2–3 comparison datasets; tracking (MMFs, granules) is deliberately
not part of this yet.

## pyMilne on Apple silicon

pyMilne lives in `~/Documents/GitHub/pyMilne` (or set `PYMILNE_DIR`). Its
`setup.py` does not build on this Mac; use the arm64 build file there:

```bash
conda activate Milne
cd ~/Documents/GitHub/pyMilne && python setup_macos_arm64.py
```

It leaves `pyMilne.cpython-311-darwin.so` next to `MilneEddington.py`. The
build file documents every deviation from `setup.py` (Apple clang +
libomp, C++17, three `std::abs<T>` calls and three dynamic OpenMP
schedules patched in a build copy, Eigen threading off). Checked on
pyMilne's own `example_CRISP` (same maps as the reference) and for
1- vs 8-thread identity.

## Run

```bash
conda activate Milne
python sst/me_invert.py nb_6302_2022-05-16T08:28:21_..._im.fits --out me_AR13010
python sst/me_quicklook.py me_AR13010 --movie            # + --scan N, --all-overviews
```

Useful options of `me_invert.py`: `--scans 0-99`, `--crop x0 x1 y0 y1`,
`--bin 2`, `--nthreads 8`, `--no-profile` (skip the CRISP transmission
profile: faster, less exact), `--overwrite`. Existing scans are skipped,
so an interrupted run resumes.

What it does per scan: read (memory-mapped) → normalise to the quiet-Sun
intensity at the far-wing point → noise in Q, U, V measured in the quiet
Sun → observed points on a regular 10 mÅ grid convolved with the CRISP
dual-FPI profile (as in pyMilne's `example_CRISP`) → inversion in row
blocks (first scan: two passes, 4 randomisations; later scans start from
the previous scan) → re-inversion of outlier pixels.

Output `me_scanNNNN.fits`: 15 planes — B [G], inc, azi [rad], vlos
[km/s], vdop [Å], eta0, damp, S0, S1, chi2, I_wing (normalised far-wing
intensity), B_los, B_trans [G], inc, azi [deg]. Header: time, mu, pointing,
plate scale, normalisation, noise, settings, `VQS` (quiet-Sun median vlos,
not subtracted).

Speed (M-series, 8 threads, CRISP profile on): ~3400 px/s for scans
started from the previous one, i.e. ~5 min per scan for a 1000×1000 FOV;
the first scan takes ~4x longer.

## Caveats

- v_los has no absolute zero point; plots show it relative to the
  quiet-Sun median of each scan.
- Azimuth and image axes are in the cube's image frame.
- Where linear polarisation is weak compared with the noise, ME can fall
  into a longitudinal-field solution (inclination 0/180°, B_trans ≈ 0,
  small Doppler width) in coherent patches; the outlier pass only fixes
  isolated pixels. pyMilne's spatially regularised inversion is the
  remedy if these patches matter.
- Tested on an SSTRED-layout cube built from the prepared AR13010 maps
  (2022-05-16 08:28:21, frames 0/11/19; 11 tunings −280…+160 mÅ), not yet
  on an original SSTRED 6302 cube.
