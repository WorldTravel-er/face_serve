#!/usr/bin/env python3
"""Create a timestamped backup of the SQLite database and data files."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def backup_data(data_dir: Path, backup_root: Path, label: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_root / f"{stamp}-{label or 'manual'}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)

    database = data_dir / "face_api.db"
    if database.exists():
        with sqlite3.connect(database) as source, sqlite3.connect(destination / "face_api.db") as target:
            source.backup(target)

    copied: list[str] = []
    for name in ("subjects", "video_analysis", "videos"):
        source_dir = data_dir / name
        if source_dir.is_dir():
            shutil.copytree(source_dir, destination / name)
            copied.append(name)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(data_dir),
        "database": database.exists(),
        "directories": copied,
        "label": label,
    }
    (destination / "backup-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/var/lib/face-serve"))
    parser.add_argument("--backup-root", type=Path, default=Path("/var/backups/face-serve"))
    parser.add_argument("--label")
    args = parser.parse_args(argv)
    destination = backup_data(args.data_dir, args.backup_root, args.label)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
