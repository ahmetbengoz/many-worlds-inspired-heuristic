import numpy as np

def tour_length(tour, D):
    """
    Computes total tour length for a given permutation.
    """
    n = len(tour)
    return sum(D[tour[i], tour[(i + 1) % n]] for i in range(n))


def random_tour(n):
    """
    Generates a random TSP tour.
    """
    tour = np.arange(n)
    np.random.shuffle(tour)
    return tour

