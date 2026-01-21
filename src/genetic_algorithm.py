import numpy as np
from .utils import tour_length, random_tour

def genetic_algorithm(D, iterations=800, population_size=30):
    """
    Simple GA-like baseline: truncation selection + swap mutation.
    Returns best-so-far tour length history.
    """
    n = D.shape[0]
    population = [random_tour(n) for _ in range(population_size)]
    history = []

    for _ in range(iterations):
        population = sorted(population, key=lambda t: tour_length(t, D))
        survivors = population[: population_size // 2]

        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent = survivors[np.random.randint(len(survivors))].copy()
            i, j = np.random.choice(n, 2, replace=False)
            parent[i], parent[j] = parent[j], parent[i]
            new_population.append(parent)

        population = new_population
        history.append(tour_length(population[0], D))

    return history
