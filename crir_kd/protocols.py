"""Canonical controlled-setting fingerprints and explicit protocol differences."""

import hashlib
import json


REQUIRED_SECTIONS = ("dataset", "student", "teacher", "optimization", "augmentation", "evaluation", "kd_features")


def protocol_fingerprint(protocol):
    if not isinstance(protocol, dict) or any(not isinstance(protocol.get(section), dict) for section in REQUIRED_SECTIONS):
        raise ValueError("protocol requires dataset, student, teacher, optimization, augmentation, evaluation and kd_features objects")
    if "method" in protocol or "seed" in protocol:
        raise ValueError("method and seed belong to run records, not controlled protocol settings")
    for section in ("student", "teacher"):
        size = protocol[section].get("input_size")
        if not isinstance(size, list) or len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size):
            raise ValueError(f"{section}.input_size must be [positive width, positive height]")
    for section, key in (("dataset", "version"), ("dataset", "split_id"), ("student", "implementation"),
                         ("teacher", "checkpoint_id"), ("evaluation", "implementation")):
        value = protocol[section].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{key} must be a nonempty string")
    try:
        canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError(f"protocol must contain finite JSON values: {error}") from error
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compare_protocols(left, right):
    left_hash, right_hash = protocol_fingerprint(left), protocol_fingerprint(right)
    differences = []
    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append({"path": path + key, "left": a.get(key), "right": b.get(key), "presence_differs": True})
                else:
                    compare(a[key], b[key], path + key + ".")
        elif type(a) is not type(b) or a != b:
            differences.append({"path": path.rstrip("."), "left": a, "right": b})
    compare(left, right, "")
    return {"comparable": left_hash == right_hash, "left_fingerprint": left_hash, "right_fingerprint": right_hash, "differences": differences}
