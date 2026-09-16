"""Pair repeated runs by seed and protocol; report descriptive deltas only."""

import statistics
from .summaries import summarize_runs
from .boxes import finite_number


def paired_results(runs, baseline, candidate, metric):
    summarize_runs(runs)
    if baseline == candidate:
        raise ValueError("baseline and candidate must differ")
    selected = [run for run in runs if run["method"] in (baseline, candidate)]
    if not selected or not {baseline, candidate} <= {run["method"] for run in selected}:
        raise ValueError("both requested methods must be present")
    protocols = {(run["protocol_fingerprint"].lower(), run["synthetic"]) for run in selected}
    if len(protocols) != 1:
        raise ValueError("paired comparison requires one protocol and one synthetic/real status")
    groups = {method: {run["seed"]: run for run in selected if run["method"] == method} for method in (baseline, candidate)}
    seeds = sorted(set(groups[baseline]) & set(groups[candidate]))
    if not seeds:
        raise ValueError("requested methods have no shared seeds")
    deltas = []
    for seed in seeds:
        if any(metric not in groups[method][seed]["metrics"] for method in (baseline, candidate)):
            raise ValueError("requested metric is missing")
        delta = groups[candidate][seed]["metrics"][metric] - groups[baseline][seed]["metrics"][metric]
        deltas.append({"seed": seed, "delta": finite_number(delta, "paired delta")})
    values = [row["delta"] for row in deltas]
    return {"baseline": baseline, "candidate": candidate, "metric": metric, "paired": deltas,
            "mean_delta": statistics.mean(values), "sample_std_delta": statistics.stdev(values) if len(values) > 1 else None,
            "unpaired_seeds": {method: sorted(set(groups[method]) - set(seeds)) for method in (baseline, candidate)},
            "synthetic": selected[0]["synthetic"], "interpretation": "Candidate minus baseline; direction depends on the metric. No significance or causal claim."}
