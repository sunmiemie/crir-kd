"""Aligned feature MSE reference with explicit optional whole-region L2 norm."""

import math
from .boxes import finite_number
from .sampling import feature_shape, sample_box


def feature_mse(student, teacher, normalize=False):
    if feature_shape(student) != feature_shape(teacher):
        raise ValueError("student and teacher sampled feature shapes must match")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be boolean")
    vectors = [[float(value) for channel in array for row in channel for value in row] for array in (student, teacher)]
    if normalize:
        for index, vector in enumerate(vectors):
            scale = max(abs(value) for value in vector)
            if scale:
                scaled = [value / scale for value in vector]
                norm = math.sqrt(math.fsum(value * value for value in scaled))
                vectors[index] = [value / norm for value in scaled]
    try:
        loss = math.fsum((left - right) ** 2 / len(vectors[0]) for left, right in zip(*vectors))
    except OverflowError:
        raise ValueError("feature MSE exceeds numeric range") from None
    return finite_number(loss, "feature MSE")


def local_feature_losses(student, teacher, student_boxes, teacher_boxes, output_size=(2, 2), normalize=False):
    if len(student_boxes) != len(teacher_boxes):
        raise ValueError("student and teacher must have the same GT instance count")
    return [feature_mse(sample_box(student, sb, output_size), sample_box(teacher, tb, output_size), normalize)
            for sb, tb in zip(student_boxes, teacher_boxes)]
