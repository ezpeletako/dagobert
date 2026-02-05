import numpy as np
from dataclasses import dataclass
from .replicator import Replicator

@dataclass(frozen=True)
class VectorReplicator(Replicator):
    genome: np.ndarray

    def express(self):
        return self.genome

    def evaluate(self) -> float:
        # simple convex fitness: max at zero
        return float(-np.sum(self.genome ** 2))

    def mutate(self, rng: np.random.Generator, sigma: float = 0.1):
        child = self.genome + rng.normal(0, sigma, size=self.genome.shape)
        return VectorReplicator(child)

