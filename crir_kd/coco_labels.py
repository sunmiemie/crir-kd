"""Structural COCO-style box audit; not an official dataset evaluator."""

from .boxes import Box, finite_number


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def inspect_coco(document):
    if not isinstance(document, dict):
        raise ValueError("COCO document must be an object")
    for key in ("images", "categories", "annotations"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"{key} must be an array")
    issues, images, categories = [], {}, set()

    def issue(section, index, error):
        issues.append({"section": section, "index": index, "severity": "error", "message": str(error)})

    for index, row in enumerate(document["images"]):
        try:
            if not isinstance(row, dict):
                raise ValueError("image must be an object")
            image_id = _integer(row.get("id"), "image id")
            if image_id in images:
                raise ValueError("duplicate image id")
            width, height = _integer(row.get("width"), "width"), _integer(row.get("height"), "height")
            if min(width, height) <= 0 or not isinstance(row.get("file_name"), str) or not row["file_name"].strip():
                raise ValueError("image requires positive dimensions and a file_name")
            images[image_id] = (width, height)
        except ValueError as error:
            issue("images", index, error)
    for index, row in enumerate(document["categories"]):
        try:
            if not isinstance(row, dict):
                raise ValueError("category must be an object")
            category = _integer(row.get("id"), "category id")
            if category in categories:
                raise ValueError("duplicate category id")
            categories.add(category)
        except ValueError as error:
            issue("categories", index, error)
    annotation_ids, valid_annotations = set(), 0
    for index, row in enumerate(document["annotations"]):
        try:
            if not isinstance(row, dict):
                raise ValueError("annotation must be an object")
            annotation_id = _integer(row.get("id"), "annotation id")
            if annotation_id in annotation_ids:
                raise ValueError("duplicate annotation id")
            annotation_ids.add(annotation_id)
            image_id = _integer(row.get("image_id"), "image_id")
            category_id = _integer(row.get("category_id"), "category_id")
            if image_id not in images or category_id not in categories:
                raise ValueError("annotation references an unknown image or category")
            coordinates = row.get("bbox")
            if not isinstance(coordinates, list) or len(coordinates) != 4:
                raise ValueError("bbox must contain [x, y, width, height]")
            box = Box.from_xywh(*coordinates)
            width, height = images[image_id]
            if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
                raise ValueError("bbox extends outside its image; inspect dataset rules before converting")
            if "area" in row and finite_number(row["area"], "area") < 0:
                raise ValueError("area must be nonnegative; segmentation area need not equal box area")
            if "iscrowd" in row and (_integer(row["iscrowd"], "iscrowd") not in (0, 1)):
                raise ValueError("iscrowd must be 0 or 1")
            valid_annotations += 1
        except ValueError as error:
            issue("annotations", index, error)
    return {"valid": not issues, "image_count": len(images), "category_count": len(categories),
            "valid_annotation_count": valid_annotations, "issues": issues}
