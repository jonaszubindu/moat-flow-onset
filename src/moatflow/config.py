"""Locate the project root, then load config.yaml from it.

The root holds `catalog/events.yaml` and `config.yaml`. It is found in
this order:

1. ``$MOATFLOW_ROOT`` if set — the explicit answer, useful for cron and
   batch jobs that start in an arbitrary directory;
2. the nearest ancestor of the working directory that contains
   `catalog/events.yaml` — covers running the scripts from anywhere in
   a checkout;
3. the package's own location, three levels up — only correct for an
   editable install (`pip install -e .`), where the package still lives
   in `<root>/src/moatflow`.

Step 3 alone was the original implementation, and it silently pointed
into `site-packages/../..` for a REGULAR install, which is how a cluster
deployment ends up looking for `config.example.yaml` inside
`.venv/lib/python3.x/`.
"""

import os
from pathlib import Path

import yaml

MARKER = Path("catalog") / "events.yaml"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Project root, or the package-relative guess if nothing matches."""
    env = os.environ.get("MOATFLOW_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    start = Path(start).resolve() if start else Path.cwd().resolve()
    for d in (start, *start.parents):
        if (d / MARKER).exists():
            return d

    pkg_guess = Path(__file__).resolve().parents[2]
    return pkg_guess          # correct for an editable install


REPO_ROOT = find_repo_root()


def load_config(path: str | Path | None = None) -> dict:
    root = REPO_ROOT
    if path is None:
        path = root / "config.yaml"
        if not path.exists():
            path = root / "config.example.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No config found at {path}.\n"
            f"  Project root resolved to: {root}\n"
            "  It should contain catalog/events.yaml and config.yaml. Fix by\n"
            "  either running the scripts from inside the checkout, or\n"
            "  setting MOATFLOW_ROOT=/path/to/moat-flow-onset, or installing\n"
            "  editable with  pip install -e .\n"
            "  (config.yaml is gitignored — create it with\n"
            "   cp config.example.yaml config.yaml  and set data_root.)")

    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = path
    cfg["repo_root"] = root
    data_root = Path(cfg["data_root"]).expanduser()
    cfg["data_root"] = data_root if data_root.is_absolute() \
        else (root / data_root).resolve()
    return cfg


def event_dir(cfg: dict, event_id: str) -> Path:
    d = cfg["data_root"] / event_id
    d.mkdir(parents=True, exist_ok=True)
    return d
