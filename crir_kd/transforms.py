"""Explicit resize/padding metadata shared by boxes and later image adapters."""

from dataclasses import dataclass

from .boxes import Box, finite_number


@dataclass(frozen=True)
class ViewTransform:
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    view_width: float
    flip_horizontal: bool = False

    def __post_init__(self):
        for field in ("scale_x", "scale_y", "pad_x", "pad_y", "view_width"):
            object.__setattr__(self, field, finite_number(getattr(self, field), field))
        if min(self.scale_x, self.scale_y, self.view_width) <= 0:
            raise ValueError("scales and view width must be positive")
        if not isinstance(self.flip_horizontal, bool):
            raise ValueError("flip_horizontal must be boolean")

    @classmethod
    def letterbox(cls, source_size, target_size, flip_horizontal=False):
        """Sizes are (width, height); uses continuous centered padding, no rounding.

        An actual image adapter must supply its rounded scales/padding explicitly
        when its pixel resize convention differs from this ideal geometry.
        """
        sw, sh = map(finite_number, source_size)
        tw, th = map(finite_number, target_size)
        if min(sw, sh, tw, th) <= 0:
            raise ValueError("image dimensions must be positive")
        scale = min(tw / sw, th / sh)
        return cls(scale, scale, (tw - sw * scale) / 2, (th - sh * scale) / 2, tw, flip_horizontal)

    def forward(self, box):
        x1, x2 = box.x1 * self.scale_x + self.pad_x, box.x2 * self.scale_x + self.pad_x
        if self.flip_horizontal:
            x1, x2 = self.view_width - x2, self.view_width - x1
        return Box(x1, box.y1 * self.scale_y + self.pad_y, x2, box.y2 * self.scale_y + self.pad_y)

    def inverse(self, box):
        x1, x2 = box.x1, box.x2
        if self.flip_horizontal:
            x1, x2 = self.view_width - x2, self.view_width - x1
        return Box((x1 - self.pad_x) / self.scale_x, (box.y1 - self.pad_y) / self.scale_y,
                   (x2 - self.pad_x) / self.scale_x, (box.y2 - self.pad_y) / self.scale_y)

    def to_feature(self, box, stride):
        stride = finite_number(stride, "stride")
        if stride <= 0:
            raise ValueError("stride must be positive")
        mapped = self.forward(box)
        return Box(mapped.x1 / stride, mapped.y1 / stride, mapped.x2 / stride, mapped.y2 / stride)
