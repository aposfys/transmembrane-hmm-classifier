"""Regenerate the pinned UniProt snapshot that backs the offline fallback.

The snapshot exists so an outage cannot stop a run, not as a substitute for
UniProt. Refresh it deliberately -- every refresh changes the dataset any
fallback run will produce, so it belongs in its own commit with the record
counts in the message.

    python -m tmclass.refresh_snapshot
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json

from . import data


def refresh(classes: tuple[str, ...] = tuple(data.QUERIES)) -> dict:
    data.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "date": dt.date.today().isoformat(),
        "source": data.UNIPROT_SEARCH,
        "fields": data.FIELDS,
        "classes": {},
    }

    for name in classes:
        print(f"{name}: downloading...", flush=True)
        payload = data._fetch_paginated(name)
        rows = payload.decode("utf-8").count("\n") - 1
        (data.SNAPSHOT_DIR / f"{name}.tsv.gz").write_bytes(gzip.compress(payload, 9))
        manifest["classes"][name] = {
            "query": data.QUERIES[name],
            "records": rows,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        print(f"{name}: {rows} records", flush=True)

    (data.SNAPSHOT_DIR / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(data.QUERIES),
        choices=list(data.QUERIES),
        help="Refresh only these classes. Defaults to all four.",
    )
    args = parser.parse_args(argv)

    manifest = refresh(tuple(args.classes))
    total = sum(entry["records"] for entry in manifest["classes"].values())
    print(f"\nSnapshot dated {manifest['date']}: {total} records across {len(args.classes)}")
    print(f"Written to {data.SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
