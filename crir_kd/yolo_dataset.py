"""Audit explicit image/label pairs, including empty negative-image labels."""

from collections import Counter

from .image_headers import image_dimensions
from .local_data import dataset_path
from .yolo_labels import inspect_yolo_labels


def inspect_yolo_dataset(root, pairs, class_count):
    inspect_yolo_labels("", class_count)
    if not isinstance(pairs, list):
        raise ValueError("pairs must be an array of image and label paths")
    reports, issues, classes, seen = [], [], Counter(), set()
    for index, pair in enumerate(pairs):
        try:
            if not isinstance(pair, dict) or not all(isinstance(pair.get(key), str) for key in ("image", "label")):
                raise ValueError("each pair requires image and label paths")
            image_path = dataset_path(root, pair["image"])
            if image_path in seen:
                raise ValueError("image appears more than once in the pair manifest")
            seen.add(image_path)
            dimensions = image_dimensions(image_path)
            labels = inspect_yolo_labels(dataset_path(root, pair["label"]).read_text(encoding="utf-8-sig"), class_count)
            for item in labels["issues"]:
                issues.append({"pair": index, **item})
            boxes = []
            for label in labels["labels"]:
                box = label["box"]
                width, height = dimensions["width"], dimensions["height"]
                boxes.append({"class_id": label["class_id"], "xywh": [box.x1 * width, box.y1 * height, box.width * width, box.height * height]})
                classes[label["class_id"]] += 1
            reports.append({"image": pair["image"], **dimensions, "boxes": boxes, "negative_image": not boxes and labels["valid"]})
        except (OSError, UnicodeError, ValueError) as error:
            issues.append({"pair": index, "severity": "error", "message": str(error)})
    return {"valid": not any(issue["severity"] == "error" for issue in issues), "images": reports,
            "class_counts": dict(sorted(classes.items())), "issues": issues}
