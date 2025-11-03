import numpy as np


class Mutation:
    """
    Base class for mutation operators.
    """
    def __init__(self):
        pass

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        pass


class MutationUniform(Mutation):
    """
    Uniform mutation operator.
    """
    def __init__(self, mutation_rate: float):
        super().__init__()
        self.mutation_rate = mutation_rate

    def mutate(self, p: np.ndarray, mutate_range: tuple, rng=np.random.default_rng()) -> np.ndarray:
        dim = len(p)
        c = p.copy()
        lb, ub = mutate_range
        for i in range(dim):
            if rng.random() < self.mutation_rate:
                c[i] = rng.uniform(lb[i], ub[i])
        return c
