"""Continuous pixel-coordinate boxes; no inclusive '+1' pixel convention."""

from dataclasses import dataclass
import math


def finite_number(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def probability(value, name="probability"):
    value = finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, finite_number(getattr(self, name), name))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValueError("box area must be positive and representable")

    @classmethod
    def from_xywh(cls, x, y, width, height):
        x, y, width, height = [finite_number(v) for v in (x, y, width, height)]
        if width <= 0 or height <= 0:
            raise ValueError("box width and height must be positive")
        return cls(x, y, x + width, y + height)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)

    def as_xywh(self):
        return (self.x1, self.y1, self.width, self.height)

    def clip(self, width, height):
        width, height = finite_number(width), finite_number(height)
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        x1, x2 = max(0, self.x1), min(width, self.x2)
        y1, y2 = max(0, self.y1), min(height, self.y2)
        return None if x2 <= x1 or y2 <= y1 else Box(x1, y1, x2, y2)


def iou(left, right):
    intersection = max(0, min(left.x2, right.x2) - max(left.x1, right.x1)) * max(
        0, min(left.y2, right.y2) - max(left.y1, right.y1)
    )
    return intersection / (left.area + right.area - intersection)
