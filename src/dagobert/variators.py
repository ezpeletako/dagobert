"""Mutation and crossover operations."""
import numpy as np
from typing import Tuple, Optional


def mutate_only(parent, rng, **kwargs):
    """Single-parent variation via mutation."""
    return parent.mutate(rng, **kwargs)


def crossover_or_mutate(a, b, rng, crossover_rate: float = 0.7, **kwargs):
    """Two-parent recombination or fallback to mutation."""
    if type(a) != type(b):
        raise TypeError("Crossover requires same-type replicators")
        
    if hasattr(type(a), 'crossover') and rng.random() < crossover_rate:
        return type(a).crossover(a, b, rng, **kwargs)
    else:
        return mutate_only(a, rng, **kwargs)


def elitism_preserve(sorted_indices: list, pop_size: int, elite_frac: float) -> int:
    """Return count of individuals to preserve unchanged."""
    return max(1, int(elite_frac * pop_size))