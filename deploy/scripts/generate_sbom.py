#!/usr/bin/env python3
"""Generate a compact SPDX 2.3 SBOM and third-party license inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spdx_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9.-]+", "-", value).strip("-")
    return f"SPDXRef-{normalized or 'package'}"


def wheel_metadata(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    license_value = (
        metadata.get("License-Expression")
        or metadata.get("License")
        or "NOASSERTION"
    ).strip()
    if not license_value or license_value.upper() == "UNKNOWN":
        license_value = "NOASSERTION"
    return {
        "name": metadata.get("Name", path.stem),
        "version": metadata.get("Version", "NOASSERTION"),
        "license": license_value,
        "summary": metadata.get("Summary", ""),
        "home_page": metadata.get("Home-page", ""),
    }


def generate(package_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    packages = []
    relationships = []
    license_inventory = []

    source_id = spdx_id(f"face-serve-{manifest['version']}")
    packages.append(
        {
            "SPDXID": source_id,
            "name": "face-serve",
            "versionInfo": manifest["version"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "supplier": "NOASSERTION",
        }
    )
    relationships.append(
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": source_id,
        }
    )

    for wheel in sorted((package_dir / "packages" / "wheelhouse").glob("*.whl")):
        metadata = wheel_metadata(wheel)
        identifier = spdx_id(f"pypi-{metadata['name']}-{metadata['version']}")
        packages.append(
            {
                "SPDXID": identifier,
                "name": metadata["name"],
                "versionInfo": metadata["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": metadata["license"],
                "checksums": [{"algorithm": "SHA256", "checksumValue": sha256_file(wheel)}],
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:pypi/{metadata['name']}@{metadata['version']}",
                    }
                ],
            }
        )
        relationships.append(
            {
                "spdxElementId": source_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": identifier,
            }
        )
        license_inventory.append(
            {
                "name": metadata["name"],
                "version": metadata["version"],
                "license": metadata["license"],
                "summary": metadata["summary"],
                "home_page": metadata["home_page"],
                "artifact": wheel.name,
            }
        )

    for model in sorted((package_dir / "models").rglob("*")):
        if not model.is_file():
            continue
        relative = model.relative_to(package_dir).as_posix()
        identifier = spdx_id(f"model-{relative}")
        packages.append(
            {
                "SPDXID": identifier,
                "name": relative,
                "versionInfo": sha256_file(model)[:12],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "checksums": [{"algorithm": "SHA256", "checksumValue": sha256_file(model)}],
                "comment": "Model redistribution rights must be reviewed before public release.",
            }
        )
        relationships.append(
            {
                "spdxElementId": source_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": identifier,
            }
        )

    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"face-serve-{manifest['version']}-{manifest['profile']}",
        "documentNamespace": (
            f"https://github.com/WorldTravler/face_serve/"
            f"sbom/{manifest['version']}/{manifest['profile']}"
        ),
        "creationInfo": {
            "created": manifest["created_at"],
            "creators": ["Tool: face-serve-generate-sbom"],
        },
        "packages": packages,
        "relationships": relationships,
    }

    sbom_path = output_dir / "face-serve.spdx.json"
    license_path = output_dir / "third-party-licenses.json"
    sbom_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    license_path.write_text(
        json.dumps(license_inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sbom_path, license_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    sbom, licenses = generate(args.package_dir, args.output_dir)
    print(sbom)
    print(licenses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
