"""Runnable supplied-prediction/feature prototype, with no model or autograd."""

from .boxes import Box
from .controls import reliability_controls
from .crops import CropView
from .feature_loss import local_feature_losses
from .matching import Instance, Prediction
from .multiview import analyze_views
from .weighting import total_loss


def _box(values):
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("box coordinates must be a four-value xyxy array")
    return Box(*values)


def _instances(rows, predictions=False):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("instances must be an array of objects")
    return [(Prediction(_box(row.get("box")), row.get("class_id"), row.get("score")) if predictions
             else Instance(_box(row.get("box")), row.get("class_id"))) for row in rows]


def reference_pipeline(document):
    if not isinstance(document, dict) or not isinstance(document.get("synthetic"), bool):
        raise ValueError("reference input must explicitly label synthetic true or false")
    gt = _instances(document.get("ground_truth"))
    predictions = _instances(document.get("reference_predictions"), True)
    views = []
    if not isinstance(document.get("views"), list):
        raise ValueError("views must be an array")
    for row in document["views"]:
        if not isinstance(row, dict) or not isinstance(row.get("size"), list) or len(row["size"]) != 2:
            raise ValueError("view requires [width,height] size")
        views.append({"id": row.get("id"), "view": CropView(_box(row.get("crop")), *row["size"], row.get("flip_horizontal", False)),
                      "predictions": _instances(row.get("predictions"), True)})
    report = analyze_views(gt, predictions, views, min_iou=document.get("min_iou", .5),
                           minimum_visibility=document.get("minimum_visibility", 1), tau=document.get("tau", 1))
    controls = reliability_controls(report, document.get("floor", 0))
    features = document.get("features")
    if not isinstance(features, dict) or not all(isinstance(features.get(key), list) for key in ("student_boxes", "teacher_boxes")):
        raise ValueError("features require student_boxes and teacher_boxes arrays")
    if len(features["student_boxes"]) != len(gt) or len(features["teacher_boxes"]) != len(gt):
        raise ValueError("feature boxes must retain every GT instance in order")
    losses = local_feature_losses(features.get("student"), features.get("teacher"),
                                  [_box(box) for box in features["student_boxes"]], [_box(box) for box in features["teacher_boxes"]],
                                  normalize=features.get("normalize", False))
    totals = {name: total_loss(document.get("detection_loss"), losses, weights, document.get("kd_strength"))
              for name, weights in controls["weights"].items()}
    return {"synthetic": document["synthetic"], "training_implemented": False, "reliability": report,
            "controls": controls, "instance_feature_losses": losses, "reference_total_losses": totals,
            "scope": "Arithmetic on supplied boxes and features. No inference, gradient, training or accuracy measurement."}
