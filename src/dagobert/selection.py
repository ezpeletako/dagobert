import numpy as np

def tournament(population, fitnesses, rng, k: int = 3):
    idx = rng.choice(len(population), size=k, replace=False)
    best = idx[np.argmax(fitnesses[idx])]
    return population[best]

