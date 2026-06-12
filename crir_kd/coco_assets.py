"""Cross-check COCO metadata against local image headers without conversion."""

from .coco_labels import inspect_coco
from .image_headers import image_dimensions
from .local_data import dataset_path


def inspect_coco_assets(root, document):
    report = inspect_coco(document)
    if not report["valid"]:
        return report
    issues, seen = [], set()
    for image in document["images"]:
        try:
            path = dataset_path(root, image["file_name"])
            if path in seen:
                raise ValueError("multiple image records resolve to the same file")
            seen.add(path)
            header = image_dimensions(path)
            if (image["width"], image["height"]) != (header["width"], header["height"]):
                raise ValueError("declared dimensions do not match the local image header")
        except (OSError, ValueError) as error:
            issues.append({"image_id": image["id"], "file_name": image["file_name"], "message": str(error)})
    return {**report, "valid": not issues, "issues": issues, "full_image_decode_verified": False}
