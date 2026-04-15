"""Deterministic greedy one-to-one quality reference, not AP evaluation."""

from dataclasses import dataclass

from .boxes import Box, iou, probability


@dataclass(frozen=True)
class Instance:
    box: Box
    class_id: int

    def __post_init__(self):
        if not isinstance(self.box, Box):
            raise ValueError("box must be a Box")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("class_id must be a nonnegative integer")


@dataclass(frozen=True)
class Prediction(Instance):
    score: float

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "score", probability(self.score, "score"))


def match_quality(ground_truth, predictions, min_iou=0.5):
    """Sort eligible pairs by descending IoU, score, then input index.

    Score must be the score for the prediction's stated class. The prototype
    accepts only matching classes, allows unmatched GT, and never reuses a
    prediction. Greedy matching is not a global optimal assignment.
    """
    min_iou = probability(min_iou, "min_iou")
    if min_iou == 0:
        raise ValueError("min_iou must be positive to forbid nonoverlapping matches")
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            overlap = iou(gt.box, pred.box) if gt.class_id == pred.class_id else 0
            if overlap >= min_iou:
                candidates.append((-overlap, -pred.score, gt_index, pred_index))
    result = [{"gt_index": i, "prediction_index": None, "iou": 0.0, "quality": 0.0} for i in range(len(ground_truth))]
    used_gt, used_pred = set(), set()
    for negative_iou, negative_score, gt_index, pred_index in sorted(candidates):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        result[gt_index].update(prediction_index=pred_index, iou=-negative_iou, quality=negative_iou * negative_score)
    return result
