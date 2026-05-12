"""Scalar reference formulas; these do not implement autograd or training."""

import math

from .boxes import finite_number, probability


def reliability_weights(quality, stability, floor=0.0, strategy="joint"):
    if len(quality) != len(stability):
        raise ValueError("quality and stability lengths differ")
    if strategy not in ("uniform", "quality", "stability", "joint"):
        raise ValueError("strategy must be uniform, quality, stability or joint")
    floor = probability(floor, "floor")
    result = []
    for q, s in zip(quality, stability):
        q, s = probability(q, "quality"), probability(s, "stability")
        signal = {"uniform": 1.0, "quality": q, "stability": s, "joint": q * s}[strategy]
        result.append(floor + (1 - floor) * signal)
    return result


def weighted_kd_loss(instance_losses, weights):
    """mean(w_i * L_i), divided by all instances, not by sum(weights).

    Therefore an entirely unreliable batch can reduce KD's total contribution.
    Empty batches return zero. Detection supervision is handled separately.
    """
    if len(instance_losses) != len(weights):
        raise ValueError("loss and weight lengths differ")
    terms = []
    for loss, weight in zip(instance_losses, weights):
        loss, weight = finite_number(loss, "loss"), probability(weight, "weight")
        if loss < 0:
            raise ValueError("instance losses must be nonnegative")
        terms.append(loss * weight)
    if not terms:
        return 0.0
    try:
        return math.fsum(terms) / len(terms)
    except OverflowError:
        # The mean of finite nonnegative terms cannot exceed their maximum.
        # Scale only the overflow case to preserve ordinary-range arithmetic.
        scale = max(terms)
        return (math.fsum(term / scale for term in terms) / len(terms)) * scale


def total_loss(detection_loss, instance_losses, weights, kd_strength):
    detection_loss = finite_number(detection_loss, "detection_loss")
    kd_strength = finite_number(kd_strength, "kd_strength")
    if detection_loss < 0 or kd_strength < 0:
        raise ValueError("detection_loss and kd_strength must be nonnegative")
    result = detection_loss + kd_strength * weighted_kd_loss(instance_losses, weights)
    return finite_number(result, "total_loss")


def matched_mean_control(weights):
    """Uniform weights with the same batch mean for a strength-matched control."""
    weights = [probability(value, "weight") for value in weights]
    mean = math.fsum(weights) / len(weights) if weights else 0.0
    return [mean] * len(weights)
