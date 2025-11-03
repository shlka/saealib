import numpy as np

class Crossover:
    """
    Base class for crossover operators.
    """
    def __init__(self):
        pass

    def crossover(self, parent: np.ndarray) -> np.ndarray:
        pass


class CrossoverBLXAlpha(Crossover):
    """
    BLX-alpha crossover operator.
    """
    def __init__(self, crossover_rate: float, gamma: float):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.gamma = gamma

    def crossover(self, p: np.ndarray, rng=np.random.default_rng()) -> np.ndarray:
        p1 = p[0]
        p2 = p[1]
        dim = len(p1)
        alpha = rng.uniform(-self.gamma, 1 + self.gamma, size=dim)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return np.array([c1, c2])
