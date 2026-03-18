"""Validate normalized YOLO detection rows without modifying annotations."""

from .boxes import Box


def inspect_yolo_labels(text, class_count):
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count <= 0:
        raise ValueError("class_count must be a positive integer")
    labels, issues = [], []
    seen = set()
    for number, row in enumerate(text.splitlines(), 1):
        if not row.strip():
            continue
        parts = row.split()
        try:
            if len(parts) != 5:
                raise ValueError("expected class_id center_x center_y width height")
            class_id = int(parts[0])
            if str(class_id) != parts[0] or not 0 <= class_id < class_count:
                raise ValueError("class_id must be an integer in the configured class range")
            cx, cy, width, height = map(float, parts[1:])
            box = Box.from_xywh(cx - width / 2, cy - height / 2, width, height)
            if box.x1 < -1e-9 or box.y1 < -1e-9 or box.x2 > 1 + 1e-9 or box.y2 > 1 + 1e-9:
                raise ValueError("normalized box extends outside the image")
            key = (class_id, box)
            if key in seen:
                issues.append({"line": number, "severity": "warning", "message": "duplicate label"})
            seen.add(key)
            labels.append({"class_id": class_id, "box": box})
        except (ValueError, OverflowError) as error:
            issues.append({"line": number, "severity": "error", "message": str(error)})
    return {"labels": labels, "issues": issues, "valid": not any(i["severity"] == "error" for i in issues)}
