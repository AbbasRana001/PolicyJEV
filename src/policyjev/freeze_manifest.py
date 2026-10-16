"""Freeze manifest: SHA-256 hashes of the files that define the methodology.

Write it at the freeze with `python -m policyjev.freeze_manifest write`. Code that produces or
evaluates test predictions verifies only the entries it uses and refuses to run on a mismatch.
"""

import argparse
import csv
import hashlib
from collections.abc import Iterable
from pathlib import Path

from policyjev.config import REPO_ROOT, load_config

MANIFEST_RELATIVE = "docs/frozen_manifest.csv"
DISTILBERT_DIR = "artifacts/models/distilbert"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TEXT_SUFFIXES = (".csv", ".json", ".yaml", ".yml", ".md", ".txt")


class ManifestError(RuntimeError):
    """The manifest is missing, incomplete, or does not match the files on disk."""


def digest_and_size(path: Path) -> tuple[str, int]:
    """SHA-256 hex digest and hashed size of a file.

    Text files (TEXT_SUFFIXES) are hashed after replacing CRLF with LF, so a checkout with
    either line ending gives the same hash. Other files are hashed byte for byte.
    """
    path = Path(path)
    if path.suffix.lower() in TEXT_SUFFIXES:
        content = path.read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(content).hexdigest(), len(content)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file (line endings normalized for text files)."""
    return digest_and_size(path)[0]


def manifest_file(root: Path | None = None) -> Path:
    """Path of the manifest under a repository root."""
    return (root if root is not None else REPO_ROOT) / MANIFEST_RELATIVE


def relative_key(path: Path, root: Path) -> str:
    """Manifest key of a file: its POSIX path relative to the root."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def expected_files(root: Path, jev_version: str) -> tuple[list[Path], list[str]]:
    """Files that define the methodology: those present, and the names of any missing."""
    present: list[Path] = []
    missing: list[str] = []

    def add(relative: str) -> None:
        path = root / relative
        if path.is_file():
            present.append(path)
        else:
            missing.append(relative)

    question_files = sorted((root / "configs").glob("questions_v*.yaml"))
    if question_files:
        present.extend(question_files)
    else:
        missing.append("configs/questions_v*.yaml")
    for name in ("logreg_word", "distilbert", f"jev_{jev_version}", "jev_stack"):
        add(f"artifacts/calibrators/{name}.json")
    add("artifacts/metrics/validation_thresholds.csv")
    add("artifacts/models/logreg_word.joblib")
    add("artifacts/policy/policy_thresholds.csv")

    folder = root / DISTILBERT_DIR
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
    if any(p.name in WEIGHT_FILES for p in files):
        present.extend(files)
    else:
        missing.append(f"{DISTILBERT_DIR}/ (model weights)")
    return present, missing


def write_entries(paths: Iterable[Path], root: Path, target: Path) -> int:
    """Write a manifest (path, sha256, size) for the given files; returns the entry count."""
    rows = sorted((relative_key(p, root), *digest_and_size(p)) for p in paths)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["path", "sha256", "size"])
        writer.writerows(rows)
    return len(rows)


def write_manifest(root: Path, jev_version: str, allow_missing: bool = False) -> int:
    """Write docs/frozen_manifest.csv; refuses if expected files are missing unless allowed."""
    present, missing = expected_files(root, jev_version)
    if missing and not allow_missing:
        raise ManifestError(
            "cannot write the manifest, expected files are missing: " + ", ".join(missing)
        )
    return write_entries(present, root, manifest_file(root))


def read_manifest(target: Path) -> dict[str, tuple[str, int]]:
    """Read a manifest into {path: (sha256, size)}."""
    if not target.is_file():
        raise ManifestError(
            f"{target.name} not found; create it at the freeze with "
            "python -m policyjev.freeze_manifest write"
        )
    with target.open("r", encoding="utf-8", newline="") as f:
        return {row["path"]: (row["sha256"], int(row["size"])) for row in csv.DictReader(f)}


def verify_entries(
    entries: Iterable[tuple[str, Path]], target: Path | None = None
) -> None:
    """Verify (manifest key, actual file) pairs against the manifest; raises ManifestError."""
    manifest = read_manifest(target if target is not None else manifest_file())
    problems = []
    for key, path in entries:
        if key not in manifest:
            problems.append(f"{key}: not listed in the manifest")
        elif not Path(path).is_file():
            problems.append(f"{key}: file not found at {path}")
        elif sha256_of(Path(path)) != manifest[key][0]:
            problems.append(f"{key}: hash does not match the frozen manifest")
    if problems:
        raise ManifestError("frozen manifest check failed: " + "; ".join(problems))


def verify_files(
    paths: Iterable[Path], root: Path | None = None, target: Path | None = None
) -> None:
    """Verify files located under the repository root against the manifest."""
    root = root if root is not None else REPO_ROOT
    verify_entries(
        [(relative_key(Path(p), root), Path(p)) for p in paths],
        target if target is not None else manifest_file(root),
    )


def verify_model_folder(
    prefix: str, folder: Path, target: Path | None = None
) -> None:
    """Verify every manifest entry under a folder prefix against files in another location."""
    manifest = read_manifest(target if target is not None else manifest_file())
    marker = prefix.rstrip("/") + "/"
    keys = [k for k in manifest if k.startswith(marker)]
    if not keys:
        raise ManifestError(f"the manifest has no entries under {marker}")
    verify_entries([(k, Path(folder) / k[len(marker):]) for k in keys], target)


def check_all(root: Path | None = None) -> int:
    """Verify every manifest entry against the files under the root; returns the count."""
    root = root if root is not None else REPO_ROOT
    manifest = read_manifest(manifest_file(root))
    verify_entries([(k, root / k) for k in manifest], manifest_file(root))
    return len(manifest)


def main() -> None:
    """Write or check docs/frozen_manifest.csv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["write", "check"])
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "write":
            version = load_config()["jev"]["primary_question_version"]
            count = write_manifest(REPO_ROOT, version, args.allow_missing)
            print(f"wrote {count} entries to {MANIFEST_RELATIVE}")
        else:
            print(f"all {check_all()} manifest entries match")
    except ManifestError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
