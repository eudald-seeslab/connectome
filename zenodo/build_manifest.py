#!/usr/bin/env python3
"""Build integrity and inventory files for a prepared Zenodo directory."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import mimetypes
from pathlib import Path


EXCLUDED = {"MANIFEST.csv", "SHA256SUMS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_rows(path: Path) -> int | None:
    name = path.name.lower()
    if name.endswith(".csv.gz") or name.endswith(".tsv.gz"):
        opener = gzip.open
    elif name.endswith(".csv") or name.endswith(".tsv"):
        opener = open
    else:
        return None

    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def media_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz"):
        return "text/csv+gzip"
    if name.endswith(".tsv.gz"):
        return "text/tab-separated-values+gzip"
    if name.endswith(".csv"):
        return "text/csv"
    if name.endswith(".tsv"):
        return "text/tab-separated-values"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deposit_dir", type=Path)
    args = parser.parse_args()

    files = sorted(
        path
        for path in args.deposit_dir.iterdir()
        if path.is_file() and path.name not in EXCLUDED
    )
    rows = [
        {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "media_type": media_type(path),
            "data_rows": "" if (count := data_rows(path)) is None else count,
            "sha256": sha256(path),
        }
        for path in files
    ]

    manifest_path = args.deposit_dir / "MANIFEST.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename",
                "size_bytes",
                "media_type",
                "data_rows",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    checksum_files = [*files, manifest_path]
    with (args.deposit_dir / "SHA256SUMS").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for path in checksum_files:
            handle.write(f"{sha256(path)}  {path.name}\n")


if __name__ == "__main__":
    main()
