"""DeepVel interface (second, independent velocity estimate).

DeepVel (Asensio Ramos, Requerey & Vitas 2017, A&A 604, A11) infers
photospheric horizontal velocities from pairs of continuum images with a
CNN. Pretrained weights are cadence- and instrument-specific, so this
module is wired for consecutive-frame pairs of the 45 s continuum cubes.

Status: interface only. To activate:
  1. pip install -e '.[deepvel]'
  2. obtain HMI-continuum-trained weights (or retrain on MURaM/STAGGER
     degraded to HMI resolution and 45 s cadence),
  3. implement `infer_pair` around the published network definition
     (https://github.com/aasensio/deepvel).
"""

import numpy as np


def infer_pair(im1: np.ndarray, im2: np.ndarray,
               weights_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (vx, vy) in km/s for one 45 s continuum image pair."""
    raise NotImplementedError(
        "DeepVel inference not wired up yet — see module docstring.")
