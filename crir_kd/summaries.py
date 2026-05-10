"""Descriptive repeated-run summaries with explicit protocol grouping."""

from collections import defaultdict
import re
import statistics

from .boxes import finite_number


def summarize_runs(runs):
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a nonempty array")
    grouped, seen, metric_sets = defaultdict(list), set(), {}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be an object")
        method, protocol, seed = run.get("method"), run.get("protocol_fingerprint"), run.get("seed")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a nonempty string")
        if not isinstance(protocol, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", protocol):
            raise ValueError("protocol_fingerprint must be a SHA-256 identifier of controlled settings")
        protocol = protocol.lower()
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(run.get("synthetic"), bool):
            raise ValueError("each run must explicitly label synthetic data")
        key = (method, protocol, run["synthetic"])
        if (*key, seed) in seen:
            raise ValueError("duplicate seed within one method/protocol group")
        seen.add((*key, seed))
        metrics = run.get("metrics")
        if not isinstance(metrics, dict) or not metrics or any(not isinstance(name, str) or not name.strip() for name in metrics):
            raise ValueError("metrics must map named measures to finite numbers")
        values = {name: finite_number(value, name) for name, value in metrics.items()}
        if key in metric_sets and metric_sets[key] != set(values):
            raise ValueError("all seeds in a group must report the same metrics")
        metric_sets[key] = set(values)
        grouped[key].append((seed, values))
    result = []
    for (method, protocol, synthetic), rows in sorted(grouped.items()):
        metrics = {}
        for metric in sorted(rows[0][1]):
            values = [row[1][metric] for row in rows]
            metrics[metric] = {"mean": statistics.mean(values),
                               "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                               "minimum": min(values), "maximum": max(values)}
        result.append({"method": method, "protocol_fingerprint": protocol, "synthetic": synthetic,
                       "seeds": sorted(seed for seed, _ in rows), "run_count": len(rows),
                       "single_run": len(rows) == 1, "metrics": metrics})
    return {"groups": result, "interpretation": "Descriptive statistics only; no significance or method-superiority claim."}
