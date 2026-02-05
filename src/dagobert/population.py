import numpy as np
from .selection import tournament

def evolve(
    population,
    rng,
    elite_frac: float = 0.05,
    sigma: float = 0.1,
):
    fitnesses = np.array([r.evaluate() for r in population])
    n = len(population)
    n_elite = max(1, int(elite_frac * n))

    elite_idx = np.argsort(fitnesses)[-n_elite:]
    elites = [population[i] for i in elite_idx]

    offspring = []
    while len(offspring) < n - n_elite:
        parent = tournament(population, fitnesses, rng)
        child = parent.mutate(rng, sigma)
        offspring.append(child)

    return elites + offspring, fitnesses

