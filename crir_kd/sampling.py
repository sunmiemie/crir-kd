"""Small pure-Python bilinear feature reference, not optimized RoIAlign/autograd."""

import math
from .boxes import Box, finite_number


def feature_shape(features):
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("features must be nonempty CHW sequences")
    height = len(features[0])
    width = len(features[0][0]) if height else 0
    if not height or not width:
        raise ValueError("feature dimensions must be positive")
    for channel in features:
        if len(channel) != height or any(len(row) != width for row in channel):
            raise ValueError("feature channels and rows must be rectangular")
        for row in channel:
            for value in row:
                finite_number(value, "feature value")
    return len(features), height, width


def sample_box(features, box, output_size=(2, 2)):
    """One bilinear sample at each bin center; feature pixels center at i+0.5.

    Box coordinates are in feature-map pixels, output_size is (height, width),
    and out-of-center sample locations use border clamping. Boxes themselves
    must lie within the feature rectangle. This is not torchvision RoIAlign.
    """
    channels, height, width = feature_shape(features)
    if not isinstance(box, Box) or box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height:
        raise ValueError("box must lie inside the feature grid")
    if len(output_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in output_size):
        raise ValueError("output_size must contain two positive integers")
    oh, ow = output_size
    result = []
    for channel in features:
        output = []
        for row in range(oh):
            y = min(height - 1, max(0, box.y1 + (row + .5) * box.height / oh - .5))
            y0, y1 = math.floor(y), min(height - 1, math.floor(y) + 1)
            line = []
            for col in range(ow):
                x = min(width - 1, max(0, box.x1 + (col + .5) * box.width / ow - .5))
                x0, x1 = math.floor(x), min(width - 1, math.floor(x) + 1)
                wx, wy = x - x0, y - y0
                value = ((channel[y0][x0] * (1-wx) + channel[y0][x1] * wx) * (1-wy)
                         + (channel[y1][x0] * (1-wx) + channel[y1][x1] * wx) * wy)
                line.append(finite_number(value, "sampled feature"))
            output.append(line)
        result.append(output)
    return result
