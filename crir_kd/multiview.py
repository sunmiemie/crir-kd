"""GT-indexed reliability analysis across explicit transformed prediction views."""

from .boxes import probability
from .crops import CropView
from .matching import Instance, match_quality
from .stability import instance_stability


def analyze_views(ground_truth, reference_predictions, views, min_iou=.5, minimum_visibility=1.0, tau=1.0):
    minimum_visibility = probability(minimum_visibility, "minimum_visibility")
    reference = match_quality(ground_truth, reference_predictions, min_iou)
    mapped = [[] for _ in ground_truth]
    visibility = [[] for _ in ground_truth]
    qualities = [[] for _ in ground_truth]
    view_ids = []
    for item in views:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"] or item["id"] in view_ids:
            raise ValueError("views require unique nonempty string ids")
        view, predictions = item.get("view"), item.get("predictions")
        if not isinstance(view, CropView) or not isinstance(predictions, list):
            raise ValueError("each view requires CropView metadata and predictions")
        view_ids.append(item["id"])
        visible = [view.visibility(gt.box, minimum_visibility) for gt in ground_truth]
        indices = [index for index, status in enumerate(visible) if status["visible"]]
        transformed_gt = [Instance(visible[index]["box"], ground_truth[index].class_id) for index in indices]
        matches = dict(zip(indices, match_quality(transformed_gt, predictions, min_iou)))
        for index in range(len(ground_truth)):
            visibility[index].append(visible[index]["visible"])
            match = matches.get(index)
            qualities[index].append(match["quality"] if match is not None else None)
            pred_index = match["prediction_index"] if match is not None else None
            mapped[index].append(view.transform.inverse(predictions[pred_index].box) if pred_index is not None else None)
    rows = []
    for index, (gt, match) in enumerate(zip(ground_truth, reference)):
        prediction = reference_predictions[match["prediction_index"]].box if match["prediction_index"] is not None else None
        stability = instance_stability(prediction, gt.box, mapped[index], visibility[index], tau=tau)
        rows.append({"gt_index": index, "class_id": gt.class_id, "quality": match["quality"],
                     "reference_matched": prediction is not None, "view_qualities": qualities[index], **stability})
    return {"instances": rows, "view_ids": view_ids, "teacher_forward_count": len(views) + 1,
            "minimum_visibility": minimum_visibility, "min_iou": min_iou,
            "coordinate_contract": "Reference predictions and GT are original pixels; each view prediction is in its own view pixels."}
