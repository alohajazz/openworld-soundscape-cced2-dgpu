#!/usr/bin/env python3
"""Standard-library verifier for the prospective FRDR correction package."""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_below(root: Path) -> Iterable[Path]:
    if root.exists():
        yield from (path for path in root.rglob("*") if path.is_file())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_hashes(root: Path, expected: dict[str, str], label: str) -> int:
    checked = 0
    for relative, expected_hash in sorted(expected.items()):
        path = root / relative
        require(path.is_file(), f"{label} missing: {relative}")
        actual = sha256(path)
        require(actual == expected_hash, f"{label} hash mismatch: {relative}")
        checked += 1
    return checked


def reject_private_formats(root: Path, forbidden: set[str], label: str) -> None:
    private = sorted(path.relative_to(root).as_posix() for path in files_below(root)
                     if path.suffix.lower() in forbidden)
    require(not private, f"{label} contains forbidden private-format file(s): {', '.join(private)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path,
                        help="Optional approved aggregate-artifact directory to hash-check.")
    args = parser.parse_args()

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(data["schema_version"] == 1, "unsupported manifest schema")
    require(data["package"] == HERE.name, "package name does not match directory")
    require(data["status"].startswith("numerical acceptance PASS"),
            "unexpected correction-freeze status")
    assertn = data["domain_assertions"]
    require(assertn["manifest_file_count"] == 50, "manifest universe must be 50 files")
    require(assertn["annotation_event_count"] == 1157, "annotation-event count must be 1,157")
    duration = Fraction(assertn["nominal_hours_fraction"]["numerator"],
                        assertn["nominal_hours_fraction"]["denominator"])
    require(duration == Fraction(451, 18), "nominal duration must remain 451/18 h")
    require("all manifest files" in assertn["annotation_completion"],
            "annotation-completion assertion absent")
    require(len(assertn["unchanged_design"]) == 4, "unchanged-design assertions incomplete")

    forbidden = set(data["forbidden_extensions"])
    reject_private_formats(HERE, forbidden, "package")
    checked = verify_hashes(HERE, data["source_sha256"], "source package")

    approved = data.get("approved_artifact_sha256", {})
    if args.artifact_root is None:
        require(not approved, "artifact hashes are present; pass --artifact-root to verify them")
        print(f"PASS source package: {checked} hashes; no frozen output hashes declared")
        return

    require(args.artifact_root.is_dir(), f"artifact root not found: {args.artifact_root}")
    reject_private_formats(args.artifact_root, forbidden, "artifact root")
    require(approved, "no approved artifact hashes declared; freeze has not populated this manifest")
    artifact_count = verify_hashes(args.artifact_root, approved, "artifact")
    print(f"PASS source package: {checked} hashes; approved artifacts: {artifact_count} hashes")


if __name__ == "__main__":
    main()
