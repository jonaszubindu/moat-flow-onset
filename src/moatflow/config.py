"""Load config.yaml (falling back to config.example.yaml) from the repo root."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = REPO_ROOT / "config.yaml"
        if not path.exists():
            path = REPO_ROOT / "config.example.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_root"] = (REPO_ROOT / cfg["data_root"]).resolve() \
        if not Path(cfg["data_root"]).is_absolute() else Path(cfg["data_root"])
    return cfg


def event_dir(cfg: dict, event_id: str) -> Path:
    d = cfg["data_root"] / event_id
    d.mkdir(parents=True, exist_ok=True)
    return d
