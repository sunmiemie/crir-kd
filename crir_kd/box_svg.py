"""Standalone SVG coordinate checks with escaped labels and no external assets."""

from html import escape
from .boxes import Box


def box_svg(width, height, annotations, title="Annotation coordinate check"):
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, height)):
        raise ValueError("SVG dimensions must be positive integers")
    if not isinstance(annotations, (list, tuple)):
        raise ValueError("annotations must be an array of objects")
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
             f'<title>{escape(str(title))}</title>', f'<rect width="{width}" height="{height}" fill="#eef2f7"/>']
    for index, row in enumerate(annotations):
        if not isinstance(row, dict) or not isinstance(row.get("xywh"), (tuple, list)) or len(row["xywh"]) != 4:
            raise ValueError("annotations require xywh boxes")
        box = Box.from_xywh(*row["xywh"])
        if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
            raise ValueError("annotation extends outside the canvas")
        label = escape(str(row.get("label", row.get("class_id", index))))
        lines.append(f'<rect x="{box.x1:g}" y="{box.y1:g}" width="{box.width:g}" height="{box.height:g}" fill="none" stroke="#2563eb" stroke-width="1"/>')
        lines.append(f'<text x="{box.x1:g}" y="{max(10, box.y1 - 2):g}" font-size="10">{label}</text>')
    return "\n".join(lines + ["</svg>"])
