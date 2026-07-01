"""Generational evolutionary loop."""
from typing import List, Tuple, Any
import numpy as np

from .evaluators import default_evaluate
from .selectors import tournament_selection
from .variators import mutate_only, elitism_preserve


EvolvedPopulation = Tuple[List[Any], List[float]]


def evolve(
    population: List[Any],
    rng: np.random.Generator,
    elite_frac: float = 0.05,
    sigma: float = 0.1,
    evaluator=None,
    selector=None,
    mutator=None,
) -> EvolvedPopulation:
    """
    Perform one generation of evolution.
    
    Args:
        population: Current population of replicators
        rng: Random number generator
        elite_frac: Fraction of top performers preserved (0.05 = 5%)
        sigma: Mutation strength passed to .mutate()
        evaluator: Callable(pop) -> List[float]; defaults to calling .evaluate() on each
        selector: Callable(pop, fitnesses, rng) -> replicator; defaults to tournament
        mutator: Callable(replicator, rng, **kwargs) -> replicator; defaults to .mutate()
    
    Returns:
        Tuple of (new_population, fitness_scores_of_new_pop)
    """
    n = len(population)
    
    # Evaluate current generation
    if evaluator is None:
        fitnesses = default_evaluate(population)
    else:
        fitnesses = evaluator(population)
    
    fitnesses = np.array(fitnesses)
    
    # Select elites
    n_elites = elitism_preserve(None, n, elite_frac)
    elite_indices = np.argsort(fitnesses)[-n_elites:]
    elites = [population[i] for i in elite_indices]
    
    # Generate offspring
    target_offspring = n - n_elites
    offspring = []
    
    while len(offspring) < target_offspring:
        # Tournament selection
        parent_a = tournament_selection(population, fitnesses, rng)
        
        # Apply mutation operator
        if mutator is None:
            child = parent_a.mutate(rng, sigma=sigma)
        else:
            child = mutator(parent_a, rng, sigma=sigma)
        
        offspring.append(child)
    
    # Form next generation
    next_gen = elites + offspring
    
    # Re-evaluate children (elites already scored)
    new_fitnesses = [
        fitnesses[i] if i in elite_indices 
        else child.evaluate() for i, child in enumerate(next_gen)
    ]
    
    return next_gen, new_fitnesses