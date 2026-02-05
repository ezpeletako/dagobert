from .population import evolve

def evolve_generator(population, rng, generations=50, **kwargs):
    """
    Generator wrapper around evolve().
    """
    for _ in range(generations):
        population, fitnesses = evolve(population, rng, **kwargs)
        yield population, fitnesses
