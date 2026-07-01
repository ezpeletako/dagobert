"""Selection operators."""
import numpy as np


def tournament_selection(population, fitnesses, rng, k: int = 3):
    """Tournament selection: sample k, pick fittest."""
    idx = rng.choice(len(population), size=k, replace=False)
    winner_idx = idx[np.argmax(fitnesses[idx])]
    return population[winner_idx]


def roulette_selection(population, fitnesses, rng):
    """Roulette wheel proportional selection (fitness must be positive)."""
    if len(fitnesses) <= 1:
        return population[0]
    
    adjusted = np.asarray(fitnesses) - min(fitnesses) + 1e-8
    probs = adjusted / adjusted.sum()
    idx = rng.choice(len(population), p=probs)
    return population[idx]


def rank_selection(population, ranks, rng):
    """Rank-based selection (uses ordinal rather than raw fitness)."""
    probs = np.arange(1, len(ranks) + 1) ** 2  # quadratic weighting
    probs /= probs.sum()
    idx = rng.choice(len(population), p=probs)
    return population[idx]