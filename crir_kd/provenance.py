"""Validate explicitly supplied run metadata without collecting machine secrets."""

import re


REQUIRED_TEXT = ("run_id", "dataset_name", "dataset_version", "split", "config_path",
                 "student", "teacher", "python_version", "framework_version", "device",
                 "evaluation_command", "raw_log_path")


def validate_run_record(record):
    if not isinstance(record, dict):
        raise ValueError("run record must be an object")
    errors = []
    for field in REQUIRED_TEXT:
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{field} must be a nonempty string")
    for field, length in (("code_commit", 40), ("split_fingerprint", 64), ("config_sha256", 64)):
        value = record.get(field)
        if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value):
            errors.append(f"{field} must contain {length} hexadecimal characters")
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        errors.append("seed must be a nonnegative integer")
    for field in ("student_input", "teacher_input"):
        size = record.get(field)
        if not isinstance(size, list) or len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size):
            errors.append(f"{field} must be [positive width, positive height]")
    if record.get("data_use") not in ("train", "validation", "test-evaluation-only"):
        errors.append("data_use must be train, validation or test-evaluation-only")
    if not isinstance(record.get("synthetic"), bool):
        errors.append("synthetic must be explicitly true or false")
    return {"valid": not errors, "errors": errors,
            "verification_scope": "Metadata structure only; files, provenance and results are not independently verified."}
