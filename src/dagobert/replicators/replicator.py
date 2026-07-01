import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Replicator:
    genome: np.ndarray

    def express(self):
        """Genome → phenotype."""
        raise NotImplementedError

    def evaluate(self) -> float:
        """Return fitness (higher is better)."""
        raise NotImplementedError

    def mutate(self, rng: np.random.Generator):
        """Return a mutated offspring."""
        raise NotImplementedError

