#!/usr/bin/env python
"""Is JSOC accepting export requests right now?

The 45 s cutouts need JSOC's export manager, because im_patch runs
server-side. That component sometimes refuses new records with
"Cant create new export control record [status=4]" while the rest of
JSOC is healthy. Run this before starting or resuming a long download.

Exit status: 0 accepting, 1 refusing, 2 JSOC unreachable — so you can
wait for it in a loop:

    until python scripts/jsoc_status.py; do sleep 600; done
"""

import sys

import drms

from moatflow.config import load_config

WINDOW = "2012.12.07_12:00:00_TAI-2012.12.07_12:01:00_TAI"
SERIES = f"hmi.Ic_45s[{WINDOW}@45s]"


def main():
    email = load_config()["jsoc_email"]
    client = drms.Client(email=email)

    try:
        keys = client.query(SERIES, key="T_REC")
        print(f"keyword query : OK ({len(keys)} records) — JSOC reachable")
    except Exception as e:
        print(f"keyword query : FAILED ({e}) — JSOC unreachable")
        return 2

    try:
        req = client.export(SERIES + "{continuum}", method="url",
                            protocol="fits")
        status = req.status
    except Exception as e:
        print(f"export request: REFUSED ({e})")
        print("  -> downloads will fail; wait and check again")
        return 1

    if status in (0, 1, 2, 6):
        print(f"export request: ACCEPTED (status={status}) — downloads can run")
        return 0
    print(f"export request: REFUSED (status={status}) — JSOC's export manager "
          "is not creating records")
    print("  -> server-side; wait and check again. Finished chunks are kept, "
          "so re-running a download resumes where it stopped.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
