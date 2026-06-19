"""Explicit COCO-box to YOLO text conversion plan; never writes annotation files."""

from .coco_labels import inspect_coco


def coco_to_yolo(document, ignored_policy="reject"):
    if ignored_policy not in ("reject", "skip"):
        raise ValueError("ignored_policy must be reject or skip")
    report = inspect_coco(document)
    if not report["valid"]:
        raise ValueError("COCO structure failed validation; inspect before converting")
    categories = sorted(document["categories"], key=lambda item: item["id"])
    mapping = {category["id"]: index for index, category in enumerate(categories)}
    images = {image["id"]: image for image in document["images"]}
    labels = {key: [] for key in images}
    skipped = []
    for annotation in document["annotations"]:
        if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
            if ignored_policy == "reject":
                raise ValueError("Crowd/ignored regions cannot be represented by ordinary YOLO labels; choose skip explicitly")
            skipped.append(annotation["id"])
            continue
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        values = [(x + width / 2) / image["width"], (y + height / 2) / image["height"], width / image["width"], height / image["height"]]
        labels[image["id"]].append(str(mapping[annotation["category_id"]]) + " " + " ".join(f"{value:.12g}" for value in values))
    return {"class_mapping": [{"category_id": category["id"], "class_id": mapping[category["id"]], "name": category.get("name")}
                              for category in categories],
            "images": [{"image_id": image_id, "file_name": images[image_id]["file_name"], "label_text": "\n".join(rows) + ("\n" if rows else "")}
                       for image_id, rows in sorted(labels.items())],
            "ignored_policy": ignored_policy, "skipped_annotation_ids": skipped,
            "warning": "Generic box conversion only. Skipping ignored/crowd regions loses supervision semantics; retain original official evaluation annotations."}
