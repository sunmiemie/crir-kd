"""Read local dataset files under one explicit root and hash split manifests."""

import hashlib
from pathlib import Path, PurePosixPath

from .splits import inspect_splits


def dataset_path(root, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("dataset path must be a nonempty string")
    relative = relative.replace("\\", "/")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or ":" in relative:
        raise ValueError("dataset path must stay inside its root")
    root = Path(root).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("dataset path must resolve to a file inside its root")
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_splits(root, manifests):
    """Fill hashes from actual local bytes and audit duplicates without writes."""
    inspect_splits(manifests)
    hashed = {}
    for split, rows in manifests.items():
        hashed[split] = []
        for row in rows:
            digest = file_sha256(dataset_path(root, row["path"]))
            if row.get("sha256") is not None and row["sha256"].lower() != digest:
                raise ValueError(f"Stored content hash does not match {row['path']}")
            hashed[split].append({**row, "sha256": digest})
    return {"manifests": hashed, **inspect_splits(hashed)}
