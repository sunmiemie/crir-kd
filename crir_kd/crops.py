"""Crop visibility and invertible coordinate metadata; no image resampling."""

from dataclasses import dataclass
from .boxes import Box, finite_number, probability
from .transforms import ViewTransform


@dataclass(frozen=True)
class CropView:
    crop: Box
    width: float
    height: float
    flip_horizontal: bool = False

    def __post_init__(self):
        if not isinstance(self.crop, Box):
            raise ValueError("crop must be a Box in original coordinates")
        for field in ("width", "height"):
            value = finite_number(getattr(self, field), field)
            if value <= 0:
                raise ValueError("view dimensions must be positive")
            object.__setattr__(self, field, value)
        if not isinstance(self.flip_horizontal, bool):
            raise ValueError("flip_horizontal must be boolean")

    @property
    def transform(self):
        sx, sy = self.width / self.crop.width, self.height / self.crop.height
        return ViewTransform(sx, sy, -self.crop.x1 * sx, -self.crop.y1 * sy, self.width, self.flip_horizontal)

    def visibility(self, box, minimum_fraction=0.0):
        minimum_fraction = probability(minimum_fraction, "minimum_fraction")
        x1, y1 = max(box.x1, self.crop.x1), max(box.y1, self.crop.y1)
        x2, y2 = min(box.x2, self.crop.x2), min(box.y2, self.crop.y2)
        if x2 <= x1 or y2 <= y1:
            return {"visible": False, "fraction": 0.0, "box": None}
        clipped = Box(x1, y1, x2, y2)
        fraction = clipped.area / box.area
        return {"visible": fraction >= minimum_fraction, "fraction": fraction, "box": self.transform.forward(clipped)}
