import math
import logging

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
    def __init__(self, atol: float = 0.0):
        super().__init__()
        self.data["x"] = np.empty((0, 0))
        self.data["y"] = np.empty((0, ))
        self.atol = atol  # tolerance for duplicate check

    @staticmethod
    def new(x: np.ndarray, y: np.ndarray, atol: float = 0.0) -> "Archive":
        archive = Archive(atol=atol)
        archive.set("x", x)
        archive.set("y", y)
        return archive

    def add(self, x: np.ndarray, y: float) -> None:
        # duplicate check
        if np.any(np.all(np.isclose(self.data["x"], x.reshape(1, -1), atol=self.atol), axis=1)):
            # TODO: implement to match the actual evaluation count (fe) with the archive size
            # self.data["x"] = np.vstack((self.data["x"], x.reshape(1, -1)))
            # self.data["y"] = np.hstack((self.data["y"], np.inf))
            return
        self.data["x"] = np.vstack((self.data["x"], x.reshape(1, -1)))
        self.data["y"] = np.hstack((self.data["y"], y))

    def get_knn(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        dist = np.linalg.norm(self.data["x"] - x, axis=1)
        idx = np.argsort(dist)[:k]
        return self.data["x"][idx], self.data["y"][idx]
    

### Interface classes for users

class Termination:
    """
    Base class for termination conditions.
    """
    def __init__(self, **kwargs):
        self.maxparameter = {}
        for k, v in kwargs.items():
            self.maxparameter[k] = v

    def get(self, key: str) -> float | None:
        return self.maxparameter.get(key, None)
    
    def set(self, key: str, value: float) -> None:
        self.maxparameter[key] = value

    def is_terminated(self, **kwargs) -> bool:
        for k, v in kwargs.items():
            if k in self.maxparameter and v >= self.maxparameter[k]:
                return True
        return False


class Problem:
    """
    Base class for problems.
    """
    def __init__(self, func, dim: int, lb: float, ub: float):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.func = func

    def evaluate(self, x: np.ndarray) -> float:
        return self.func(x)


class Algorithm:
    """
    Base class for algorithms.
    """
    def __init__(self):
        pass

    def step(self):
        pass


class GA(Algorithm):
    """
    Genetic Algorithm class.
    """
    def __init__(self, crossover, mutation, selection):
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

    def ask(self, optimizer):
        candidate = np.empty((0, optimizer.problem.dim))
        # parent = self.selection.select_parent(optimizer.population)
        parent = optimizer.population.get("x")
        for i in range(0, len(optimizer.population), 2):
            p1 = parent[i % len(parent)]
            p2 = parent[(i + 1) % len(parent)]
            if optimizer.rng.random() < self.crossover.crossover_rate:
                c1, c2 = self.crossover.crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            candidate = np.vstack((candidate, c1, c2))
        for i in range(len(candidate)):
            candidate[i] = self.mutation.mutate(candidate[i])
        return candidate[:optimizer.popsize]
    
    def tell(self, optimizer, offspring, offspring_fit):
        # select a best solution in parent
        best_idx = np.argmin(optimizer.population.get("f"))
        parent_best = optimizer.population.get("x")[best_idx]
        parent_best_fit = optimizer.population.get("f")[best_idx]
        parent = np.delete(optimizer.population.get("x"), best_idx, axis=0)
        parent_fit = np.delete(optimizer.population.get("f"), best_idx, axis=0)
        # update population and fitness
        pop_cand = np.vstack((parent_best, parent, offspring))
        fit_cand = np.hstack((parent_best_fit, parent_fit, offspring_fit))  
        pop_cand = pop_cand[np.argsort(fit_cand)]
        fit_cand = np.sort(fit_cand)
        optimizer.population.set("x", pop_cand[:optimizer.popsize])
        optimizer.population.set("f", fit_cand[:optimizer.popsize])

    def step(self, optimizer):
        pass


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
    def __init__(self, crossover_rate: float, gamma: float, lb: float, ub: float):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.gamma = gamma
        self.lb = lb
        self.ub = ub

    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dim = len(p1)
        alpha = np.random.uniform(-self.gamma, 1 + self.gamma, size=dim)
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = (1 - alpha) * p1 + alpha * p2
        return np.clip(c1, self.lb, self.ub), np.clip(c2, self.lb, self.ub)
    

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
    def __init__(self, mutation_rate: float, lb: float, ub: float):
        super().__init__()
        self.mutation_rate = mutation_rate
        self.lb = lb
        self.ub = ub

    def mutate(self, p: np.ndarray) -> np.ndarray:
        dim = len(p)
        c = p.copy()
        for i in range(dim):
            if np.random.rand() < self.mutation_rate:
                c[i] = np.random.uniform(self.lb, self.ub)
        return c


class Selection:
    """
    Base class for selection operators.
    """
    def __init__(self):
        pass

    def select(self) -> np.ndarray:
        pass


class Surrogate:
    """
    Base class for surrogate models.
    """
    def __init__(self):
        pass

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        pass

    def predict(self, test_x: np.ndarray) -> float:
        pass


def gaussian_kernel(x1, x2, sigma=2.0):
    return math.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))


class RBFsurrogate(Surrogate):
    def __init__(self, kernel, dim):
        self.dim = dim
        self.train_x = []
        self.train_y = []
        self.kernel = kernel
        self.weights = []
        self.kernel_matrix = []

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        n_samples = len(train_x)
        self.kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.kernel_matrix[i, j] = self.kernel(train_x[i], train_x[j])
        # self.weights = np.linalg.solve(self.kernel_matrix, train_y)
        self.weights = np.linalg.solve(self.kernel_matrix, (train_y - np.mean(train_y)))

    def predict(self, test_x):
        n_samples = len(self.train_x)
        prediction = 0
        for i in range(n_samples):
            prediction += self.kernel(test_x, self.train_x[i]) * self.weights[i]
        return prediction + np.mean(self.train_y)


class ModelManager:
    """
    Base class for surrogate model manager.
    """
    def __init__(self):
        pass


class IndividualBasedStrategy(ModelManager):
    """
    Individual-based strategy for surrogate model management.
    """
    def __init__(self):
        super().__init__()
        # parameters
        self.surrogate_model = None
        self.candidate = None

        # parameters (optional)
        self.n_train = 50
        self.knn = 50
        self.rsm = 0.1

    def run_strategy(self, optimizer):
        n_cand = len(self.candidate)
        psm = int(self.rsm * n_cand)
        rbf_model = optimizer.surrogate

        self.candidate_fit = np.zeros(n_cand)

        # predict all candidates using surrogate model
        for i in range(n_cand):
            # get training data for candidate[i]
            train_x, train_y = optimizer.archive.get_knn(self.candidate[i], k=self.knn)
            # train RBF model
            rbf_model.fit(train_x, train_y)
            # predict candidate[i]
            self.candidate_fit[i] = rbf_model.predict(self.candidate[i].reshape(1, -1))

        # psm individuals are evaluated using the true function
        self.candidate = self.candidate[np.argsort(self.candidate_fit)]
        self.candidate_fit = np.sort(self.candidate_fit)

        self.candidate_eval = self.candidate[:psm]
        self.candidate_eval_fit = np.array([optimizer.problem.evaluate(ind) for ind in self.candidate_eval])
        self.candidate_fit[:psm] = self.candidate_eval_fit
        optimizer.fe += psm

        # add evaluated individuals to the archive
        for i in range(psm):
            optimizer.archive.add(self.candidate_eval[i], self.candidate_eval_fit[i])

        return self.candidate, self.candidate_fit

class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self):
        self.problem = None
        self.algorithm = None
        self.surrogate = None
        self.modelmanager = None
        self.termination = None

        self.archive_atol = 0.0
        self.archive = None
        self.archive_init_size = 50

        self.seed = 0
        self.rng = np.random.default_rng(seed=self.seed)

        self.fe = 0
        self.gen = 0

        self.popsize = 40

    def _initialize(self, n_init_archive: int):
        archive_x = self.rng.uniform(self.problem.lb, self.problem.ub, (n_init_archive, self.problem.dim))
        archive_y = np.array([self.problem.evaluate(ind) for ind in archive_x])
        archive_x = archive_x[np.argsort(archive_y)]
        archive_y = np.sort(archive_y)
        self.archive = Archive.new(archive_x, archive_y, atol=self.archive_atol)

        self.population = Population.new("x", self.archive.get("x")[:self.popsize])
        self.population.set("f", self.archive.get("y")[:self.popsize])

        self.fe = self.archive_init_size
        self.gen = 0

    def run(self):
        self._initialize(self.archive_init_size)
        while not self.termination.is_terminated(fe=self.fe):
            # ask
            cand = self.algorithm.ask(self)

            # surrogate
            self.modelmanager.candidate = cand
            cand, cand_fit = self.modelmanager.run_strategy(self)

            # tell
            self.algorithm.tell(self, cand, cand_fit)
            self.gen += 1

            # logging info
            logging.info(f"fbest: {self.population.get('f')[0]}, fe: {self.fe}, gen: {self.gen}")
