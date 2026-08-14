"""Compute reliability ablations from the same cached teacher-view outputs."""

import math
from .boxes import probability
from .weighting import reliability_weights, matched_mean_control


def reliability_controls(report, floor=0.0):
    floor = probability(floor, "floor")
    if not isinstance(report, dict) or not isinstance(report.get("instances"), list) or not isinstance(report.get("view_ids"), list):
        raise ValueError("expected a multiview analysis report")
    quality, stability, mean_quality = [], [], []
    for row in report["instances"]:
        q, s = probability(row["quality"], "quality"), probability(row["stability"], "stability")
        views = row["view_qualities"]
        if len(views) != len(report["view_ids"]):
            raise ValueError("view quality count differs from the view budget")
        eligible = [q] + [probability(value, "view quality") for value in views if value is not None]
        quality.append(q)
        stability.append(s)
        mean_quality.append(math.fsum(eligible) / len(eligible))
    weights = {name: reliability_weights(quality, stability, floor, name) for name in ("uniform", "quality", "stability", "joint")}
    weights["multi_view_quality"] = reliability_weights(mean_quality, stability, floor, "quality")
    weights["joint_mean_control"] = matched_mean_control(weights["joint"])
    return {"weights": weights, "mean_view_quality": mean_quality,
            "teacher_forward_count": len(report["view_ids"]) + 1,
            "note": "Controls reuse identical supplied teacher views; missing visible matches contribute zero, invisible views are excluded."}
