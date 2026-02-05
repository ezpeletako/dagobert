import numpy as np
from dagobert.vector_replicator import VectorReplicator
from dagobert.population import evolve
from dagobert.rng import make_rng

def test_fitness_improves():
    rng = make_rng(42)
    pop = [
        VectorReplicator(rng.normal(size=10))
        for _ in range(50)
    ]

    best = []
    for _ in range(30):
        pop, fitnesses = evolve(pop, rng)
        best.append(fitnesses.max())

    assert best[-1] > best[0]
