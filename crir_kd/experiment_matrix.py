"""Build explicit planned runs without launching training or fabricating results."""

import re
from .protocols import protocol_fingerprint


def experiment_matrix(protocol, methods, seeds):
    fingerprint = protocol_fingerprint(protocol)
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a nonempty array")
    names, runs = set(), []
    for method in methods:
        if not isinstance(method, dict):
            raise ValueError("method entries must be objects")
        name, forwards = method.get("name"), method.get("teacher_forwards_per_step")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name in names:
            raise ValueError("method names must be unique lowercase identifiers")
        if isinstance(forwards, bool) or not isinstance(forwards, int) or forwards < 0:
            raise ValueError("teacher_forwards_per_step must be a nonnegative integer")
        names.add(name)
        for seed in seeds:
            runs.append({"run_id": f"{name}-seed{seed}-{fingerprint[:12]}", "method": name, "seed": seed,
                         "protocol_fingerprint": fingerprint, "teacher_forwards_per_step": forwards,
                         "status": "planned-not-executed"})
    if "joint" in names and "quality" not in names:
        raise ValueError("joint reliability experiments require the quality-only baseline")
    budgets = sorted(set(run["teacher_forwards_per_step"] for run in runs))
    return {"runs": runs, "planned_run_count": len(runs), "teacher_compute_budgets": budgets,
            "compute_matched": len(budgets) <= 1, "warning": "Plans contain no measured results. Different teacher budgets require explicit compute-matched controls."}
