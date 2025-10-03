import numpy as np


class Individual:
    """
    Individual class to handle single individual in population.
    """
    def __init__(self, population: "Population", index: int):
        self.population = population
        self.index = index

    def get(self, key: str) -> np.ndarray:
        if key not in self.population.data:
            return None
        return self.population.get(key)[self.index]

    def set(self, key: str, value: np.ndarray) -> None:
        self.population.get(key)[self.index] = value


class Solution(Individual):
    """
    Solution class to generate single individual easily.
    """
    def __init__(self, x: np.ndarray, **kwargs):
        pop = Population.new("x", np.array([x]))
        for k, v in kwargs.items():
            pop.set(k, np.array([v]))
        super().__init__(pop, 0)



class Population:
    """
    Base class for population.
    (self.data must have at least "x" key.)
    """
    def __init__(self):
        self.data = {}
    
    @staticmethod
    def new(key: str, value: np.ndarray) -> "Population":
        pop = Population()
        pop.set(key, value)
        return pop

    def get(self, key: str) -> np.ndarray:
        if key not in self.data:
            return None
        return self.data.get(key)

    def set(self, key: str, value: np.ndarray) -> None:
        self.data[key] = value

    def __len__(self):
        if "x" in self.data:
            return self.data["x"].shape[0]
        return 0
    
    def __getitem__(self, index):    
        if isinstance(index, int):
            return Individual(self, index)
        elif isinstance(index, slice):
            return [Individual(self, i) for i in range(*index.indices(len(self)))]
        else:
            raise TypeError("Invalid argument type.")


class Archive(Population):
    """
    Archive class to handle archive of evaluated solutions.
    """
    def __init__(self, atol: float = 1e-8):
        super().__init__()
        self.data["x"] = np.empty((0, 0))
        self.data["y"] = np.empty((0, ))
        self.atol = atol  # tolerance for duplicate check

    @staticmethod
    def new(x: np.ndarray, y: np.ndarray, atol: float = 1e-8) -> "Archive":
        archive = Archive(atol=atol)
        archive.set("x", x)
        archive.set("y", y)
        return archive

    def add(self, x: np.ndarray, y: float) -> None:
        # duplicate check
        if np.any(np.all(np.isclose(self.data["x"], x, atol=self.atol), axis=1)):
            self.data["x"] = np.vstack((self.data["x"], None))
            self.data["y"] = np.hstack((self.data["y"], None))
            return
        self.data["x"] = np.vstack((self.data["x"], x.reshape(1, -1)))
        self.data["y"] = np.hstack((self.data["y"], y))

    def get_knn(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        dist = np.linalg.norm(self.data["x"] - x, axis=1)
        idx = np.argsort(dist)[:k]
        return self.data["x"][idx], self.data["y"][idx]