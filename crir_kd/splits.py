"""Audit supplied split manifests for exact duplicates and shared source groups."""

from collections import defaultdict
import hashlib
import json
from pathlib import PurePosixPath
import re


def inspect_splits(manifests):
    """Each split maps to rows with path, optional sha256 and source_group.

    Content hashes must be supplied by the dataset owner; paths are identifiers
    relative to a shared dataset root. This does not detect near-duplicates.
    """
    if not isinstance(manifests, dict) or not manifests:
        raise ValueError("manifests must be a nonempty object mapping split names to rows")
    if any(not isinstance(split, str) or not split.strip() for split in manifests):
        raise ValueError("split names must be nonempty strings")
    paths, hashes, groups = defaultdict(list), defaultdict(list), defaultdict(set)
    canonical, issues = {}, []
    for split, rows in sorted(manifests.items()):
        if not isinstance(split, str) or not split.strip() or not isinstance(rows, list):
            raise ValueError("split names must be strings and split rows must be arrays")
        normalized = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError(f"{split}[{index}] requires a relative path")
            raw = row["path"].replace("\\", "/")
            path = PurePosixPath(raw)
            if not raw or path.is_absolute() or ".." in path.parts or ":" in raw or str(path) == ".":
                raise ValueError(f"{split}[{index}] path must stay relative to the shared dataset root")
            entry = {"path": str(path)}
            paths[str(path)].append(split)
            digest = row.get("sha256")
            if digest is not None:
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    raise ValueError(f"{split}[{index}] sha256 must contain 64 hexadecimal characters")
                entry["sha256"] = digest.lower()
                hashes[digest.lower()].append(split)
            group = row.get("source_group")
            if group is not None:
                if not isinstance(group, str) or not group.strip():
                    raise ValueError("source_group must be a nonempty string")
                entry["source_group"] = group
                groups[group].add(split)
            normalized.append(entry)
        canonical[split] = sorted(normalized, key=lambda row: json.dumps(row, sort_keys=True))
    for kind, entries in (("duplicate_path", paths), ("duplicate_content", hashes)):
        for key, splits in sorted(entries.items()):
            if len(splits) > 1:
                issues.append({"kind": kind, "value": key, "splits": sorted(splits), "cross_split": len(set(splits)) > 1})
    for group, splits in sorted(groups.items()):
        if len(splits) > 1:
            issues.append({"kind": "shared_source_group", "value": group, "splits": sorted(splits), "cross_split": True})
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"valid": not issues, "issues": issues, "fingerprint": hashlib.sha256(payload).hexdigest(),
            "counts": {split: len(rows) for split, rows in canonical.items()},
            "limitations": "Checks supplied paths, hashes and groups only; no near-duplicate image analysis."}
