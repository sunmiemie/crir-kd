"""Visibility-aware reference stability after predictions are inverse-mapped."""

import math

from .boxes import Box, finite_number


def instance_stability(reference, normalization_box, views, visible, tau=1.0, minimum_scale=1.0):
    """Missing prediction is None; cropped-out views have visible=False.

    All boxes must already be in the same original-image coordinates. Variation
    is the mean of normalized absolute center and size shifts from the reference
    prediction. The score is coverage * exp(-variation/tau). No eligible views
    or no reference prediction yields zero, never an apparently perfect score.
    This function cannot establish whether a stable prediction is correct.
    """
    tau, minimum_scale = finite_number(tau, "tau"), finite_number(minimum_scale, "minimum_scale")
    if tau <= 0 or minimum_scale <= 0:
        raise ValueError("tau and minimum_scale must be positive")
    if len(views) != len(visible) or any(not isinstance(v, bool) for v in visible):
        raise ValueError("views and boolean visibility must have equal lengths")
    if not isinstance(normalization_box, Box) or (reference is not None and not isinstance(reference, Box)):
        raise ValueError("normalization_box and reference must be boxes; reference may be None")
    if any(box is not None and not isinstance(box, Box) for box in views):
        raise ValueError("views must contain boxes or None")
    eligible = [box for box, is_visible in zip(views, visible) if is_visible]
    matches = [box for box in eligible if box is not None]
    eligible_count, matched_count = len(eligible), len(matches)
    coverage = matched_count / eligible_count if eligible_count else 0.0
    result = {"visible_views": eligible_count, "matched_views": matched_count,
              "missing_views": eligible_count - matched_count, "coverage": coverage,
              "mean_variation": None, "stability": 0.0}
    if reference is None or not matches:
        return result
    width, height = max(normalization_box.width, minimum_scale), max(normalization_box.height, minimum_scale)
    cx, cy = reference.center
    variations = [(abs(box.center[0] - cx) / width + abs(box.center[1] - cy) / height
                   + abs(box.width - reference.width) / width + abs(box.height - reference.height) / height) / 4
                  for box in matches]
    mean_variation = math.fsum(variations) / matched_count
    result.update(mean_variation=mean_variation, stability=coverage * math.exp(-mean_variation / tau))
    return result
