"""User-defined size diagnostics, deliberately separate from official AP bins."""

import math

from .boxes import Box, finite_number, probability


def size_diagnostics(instances, boundaries, coordinate_space):
    """Bin sqrt(box area) using [lower, upper) intervals in named coordinates.

    Every row needs box, matched (bool), quality and weight. This retains
    unmatched targets in denominators; it reports teacher match coverage, not
    official recall/AP. Boundaries are explicit to avoid inventing tiny bins.
    """
    if not isinstance(coordinate_space, str) or not coordinate_space.strip():
        raise ValueError("coordinate_space must explicitly name the coordinate convention")
    boundaries = [finite_number(value, "boundary") for value in boundaries]
    if any(value <= 0 for value in boundaries) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("boundaries must be strictly increasing and positive")
    buckets = [[] for _ in range(len(boundaries) + 1)]
    for row in instances:
        if not isinstance(row, dict) or not isinstance(row.get("box"), Box) or not isinstance(row.get("matched"), bool):
            raise ValueError("each instance requires a Box and boolean matched flag")
        quality, weight = probability(row.get("quality"), "quality"), probability(row.get("weight"), "weight")
        size = math.sqrt(row["box"].area)
        index = sum(size >= boundary for boundary in boundaries)
        buckets[index].append((row["matched"], quality, weight))
    results = []
    for index, rows in enumerate(buckets):
        count = len(rows)
        missing = sum(not matched for matched, _, _ in rows)
        results.append({"lower_inclusive": 0 if index == 0 else boundaries[index - 1],
                        "upper_exclusive": boundaries[index] if index < len(boundaries) else None,
                        "count": count, "unmatched_count": missing,
                        "unmatched_rate": missing / count if count else None,
                        "mean_quality": math.fsum(q for _, q, _ in rows) / count if count else None,
                        "mean_weight": math.fsum(w for _, _, w in rows) / count if count else None})
    return {"coordinate_space": coordinate_space, "size_definition": "sqrt(box area)",
            "official_dataset_metrics": False, "groups": results}
