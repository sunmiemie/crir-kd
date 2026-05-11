"""Synthetic end-to-end arithmetic walkthrough; no detector inference."""

from .boxes import Box
from .diagnostics import size_diagnostics
from .matching import Instance, Prediction, match_quality
from .stability import instance_stability
from .transforms import ViewTransform
from .weighting import reliability_weights, total_loss


def synthetic_demo():
    ground_truth = [Instance(Box(20, 30, 24, 34), 0), Instance(Box(60, 50, 68, 58), 0)]
    teacher_view = ViewTransform.letterbox((100, 80), (200, 200), flip_horizontal=True)
    student_view = ViewTransform.letterbox((100, 80), (100, 100))
    # A toy prediction passes through a teacher view and back to original pixels.
    original_prediction = Box(20, 30, 24, 34)
    transformed_prediction = teacher_view.forward(original_prediction)
    predictions = [Prediction(teacher_view.inverse(transformed_prediction), 0, .9)]
    matches = match_quality(ground_truth, predictions)
    stability = [instance_stability(original_prediction, ground_truth[0].box,
                                    [original_prediction, None, None], [True, True, False]),
                 instance_stability(None, ground_truth[1].box, [None, None], [True, True])]
    quality = [match["quality"] for match in matches]
    weights = reliability_weights(quality, [report["stability"] for report in stability], floor=.1)
    rows = [{"box": student_view.forward(gt.box), "matched": match["prediction_index"] is not None,
             "quality": q, "weight": weight}
            for gt, match, q, weight in zip(ground_truth, matches, quality, weights)]
    return {"synthetic": True, "training_implemented": False,
            "description": "Hand-written boxes and scalar losses; not measured model performance.",
            "matches": matches, "stability": stability, "weights": weights,
            "toy_total_loss": total_loss(2.0, [.4, .8], weights, .5),
            "diagnostics": size_diagnostics(rows, [8, 16], "student input pixels (toy thresholds)")}
