from .selection import tournament

def default_operators(population, fitnesses, rng, k=3):
    def select():
        idx = rng.choice(len(population), size=k, replace=False)
        best_idx = idx[np.argmax(fitnesses[idx])]
        return population[best_idx]
    
    return {'select': select}