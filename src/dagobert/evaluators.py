"""Fitness evaluation layer."""
from typing import Protocol, List, Optional


class Evaluator(Protocol):
    """Interface for fitness calculators."""
    
    def score(self, replicator) -> float:
        ...


def default_evaluate(population) -> List[float]:
    """Simple batch evaluation using each replicate's evaluate() method."""
    return [r.evaluate() for r in population]


class CachedEvaluator:
    """Memoize expensive fitness calculations."""
    
    def __init__(self, wrapped_evaluator=None):
        self.wrapped = wrapped_evaluator or (lambda x: x.evaluate())
        self._cache = {}
    
    def score(self, replicator) -> float:
        key = id(replicator)
        if key not in self._cache:
            self._cache[key] = self.wrapped(replicator)
        return self._cache[key]
    
    def clear(self):
        self._cache.clear()