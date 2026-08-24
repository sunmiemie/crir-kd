"""Seeded size/quality-stratified weight permutation for an instance-order control."""

from bisect import bisect_right
from collections import defaultdict
import random
from .boxes import finite_number, probability


def permute_weights(weights, sizes, quality, size_edges, quality_edges, seed=0):
    if not len(weights) == len(sizes) == len(quality):
        raise ValueError("instance arrays must have equal lengths")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    weights = [probability(value, "weight") for value in weights]
    quality = [probability(value, "quality") for value in quality]
    sizes = [finite_number(value, "size") for value in sizes]
    size_edges = [finite_number(value, "size edge") for value in size_edges]
    quality_edges = [probability(value, "quality edge") for value in quality_edges]
    if any(value <= 0 for value in sizes + size_edges):
        raise ValueError("sizes and size boundaries must be positive")
    if any(a >= b for edges in (size_edges, quality_edges) for a, b in zip(edges, edges[1:])):
        raise ValueError("stratum boundaries must increase strictly")
    groups = defaultdict(list)
    for index, (size, q) in enumerate(zip(sizes, quality)):
        groups[(bisect_right(size_edges, size), bisect_right(quality_edges, q))].append(index)
    rng, source_indices = random.Random(seed), list(range(len(weights)))
    for indices in groups.values():
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for destination, source in zip(indices, shuffled):
            source_indices[destination] = source
    return {"weights": [weights[index] for index in source_indices], "source_indices": source_indices,
            "seed": seed, "stratum_sizes": [len(indices) for indices in groups.values()],
            "changed_assignments": sum(a != b for a, b in enumerate(source_indices))}
