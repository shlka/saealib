import math
import logging
from enum import Enum, auto
from collections import defaultdict

import numpy as np
import scipy


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
    def __init__(self, atol: float = 0.0, rtol: float = 0.0):
        super().__init__()
        self.data["x"] = np.empty((0, 0))
        self.data["y"] = np.empty((0, ))
        self.atol = atol  # tolerance for duplicate check
        self.rtol = rtol  # relative tolerance for duplicate check

    @staticmethod
    def new(x: np.ndarray, y: np.ndarray, atol: float = 0.0, rtol: float = 0.0) -> "Archive":
        archive = Archive(atol=atol, rtol=rtol)
        archive.set("x", x)
        archive.set("y", y)
        return archive

    def add(self, x: np.ndarray, y: float) -> None:
        # duplicate check
        if np.any(np.all(np.isclose(self.data["x"], x.reshape(1, -1), atol=self.atol, rtol=self.rtol), axis=1)):
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
    def __init__(self, func, dim: int, n_obj: int, weight: np.ndarray, lb: list[float], ub: list[float], constraints: list["Constraint"] = None, eps: float = 1e-6):
        self.dim = dim
        self.n_obj = n_obj
        self.weight = weight
        self.eps = eps
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.func = func
        if constraints is None:
            constraints_list = []
        else:
            constraints_list = constraints
        
        for i in range(dim):
            constraints_list.append(Constraint(lambda x, i=i: x[i] - self.ub[i], type=ConstraintType.INEQ))
            constraints_list.append(Constraint(lambda x, i=i: self.lb[i] - x[i], type=ConstraintType.INEQ))
        
        self.constraint_manager = ConstraintManager(constraints=constraints_list)

        #TODO: multiple objective support
        self.comparator = SingleObjectiveComparator(weight=weight, eps=eps)

    def evaluate(self, x: np.ndarray) -> float:
        return self.func(x)


class ConstraintType(Enum):
    EQ = auto()
    INEQ = auto()


class Constraint:
    def __init__(self, func, type: ConstraintType = ConstraintType.INEQ):
        self.type_constraint = type
        self.func = func

    def evaluate(self, x: np.ndarray) -> float:
        v = self.func(x)
        if self.type_constraint == ConstraintType.INEQ:
            return max(0, v)
        elif self.type_constraint == ConstraintType.EQ:
            return abs(v)


class ConstraintManager:
    def __init__(self, constraints: list[Constraint]):
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

    def evaluate(self, x: np.ndarray) -> float:
        if not self.constraints:
            return 0.0
        return sum(constraint.evaluate(x) for constraint in self.constraints)

    def evaluate_population(self, population: Population) -> np.ndarray:
        if not self.constraints:
            return np.zeros(len(population))
        return np.array([self.evaluate(ind.get("x")) for ind in population])


class Algorithm:
    """
    Base class for algorithms.
    """
    def __init__(self):
        pass

    def ask(self, optimizer) -> np.ndarray:
        pass

    def tell(self, optimizer, offspring: np.ndarray, offspring_fit: np.ndarray) -> None:
        pass

    def step(self):
        pass


class GA(Algorithm):
    """
    Genetic Algorithm class.
    """
    def __init__(self, crossover: "Crossover", mutation: "Mutation", parent_selection: "ParentSelection", survivor_selection: "SurvivorSelection"):
        super().__init__()
        self.crossover = crossover
        self.mutation = mutation
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection

    def ask(self, optimizer):
        candidate = np.empty((0, optimizer.problem.dim))
        popsize = len(optimizer.population)
        pop = optimizer.population.get("x")
        n_pair = math.ceil(popsize / 2)
        parent_idx_m = self.parent_selection.select(
            optimizer,
            pop,
            optimizer.population.get("f"),
            np.zeros(popsize),# TODO: use cv if constraints are defined
            n_pair=n_pair,
            n_parents=2,# TODO: recieve n_parents from Crossover
            rng=optimizer.rng)
        for i in range(n_pair):
            parent = pop[parent_idx_m[i]]
            if optimizer.rng.random() < self.crossover.crossover_rate:
                c = self.crossover.crossover(parent, rng=optimizer.rng)
            else:
                c = parent.copy()
            candidate = np.vstack((candidate, c))
        optimizer.dispatch(CallbackEvent.POST_CROSSOVER, data=candidate)
        candidate_len = len(candidate)
        for i in range(candidate_len):
            candidate[i] = self.mutation.mutate(candidate[i], rng=optimizer.rng)
        optimizer.dispatch(CallbackEvent.POST_MUTATION, data=candidate)
        return candidate[:optimizer.popsize]
    
    def tell(self, optimizer, offspring, offspring_fit):
        cmp = optimizer.problem.comparator
        # select a best solution in parent
        best_idx = np.argmin(optimizer.population.get("f"))
        parent_best = optimizer.population.get("x")[best_idx]
        parent_best_fit = optimizer.population.get("f")[best_idx]
        parent = np.delete(optimizer.population.get("x"), best_idx, axis=0)
        parent_fit = np.delete(optimizer.population.get("f"), best_idx, axis=0)
        # update population and fitness
        pop_cand = np.vstack((parent_best, parent, offspring))
        fit_cand = np.hstack((parent_best_fit, parent_fit, offspring_fit))  
        # TODO: use cv if constraints are defined
        cand_idx = cmp.sort(fit_cand, np.zeros_like(fit_cand))
        pop_cand = pop_cand[cand_idx]
        fit_cand = fit_cand[cand_idx]
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
    def __init__(self, mutation_rate: float, lb: np.ndarray, ub: np.ndarray):
        super().__init__()
        self.mutation_rate = mutation_rate
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)

    def mutate(self, p: np.ndarray, rng=np.random.default_rng()) -> np.ndarray:
        dim = len(p)
        c = p.copy()
        for i in range(dim):
            if rng.random() < self.mutation_rate:
                c[i] = rng.uniform(self.lb[i], self.ub[i])
        return c


class ParentSelection:
    """
    Base class for parent selection operators.
    """
    def __init__(self):
        pass

    def select(self, population: Population) -> np.ndarray:
        pass


class TournamentSelection(ParentSelection):
    """
    Tournament selection operator.
    """
    def __init__(self, tournament_size: int):
        super().__init__()
        self.tournament_size = tournament_size

    def select(self, opt: "Optimizer", pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
        n_pop = len(pop_x)
        cmp = opt.problem.comparator
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        for i in range(n_pair):
            for j in range(n_parents):
                tournament_idx = rng.choice(n_pop, size=self.tournament_size, replace=False)
                best_idx = tournament_idx[0]
                for idx in tournament_idx[1:]:
                    if cmp.compare(pop_f[idx:idx+1], pop_cv[idx], pop_f[best_idx:best_idx+1], pop_cv[best_idx]) < 0:
                        best_idx = idx
                selected_idx[i, j] = best_idx
        return selected_idx


class SequentialSelection(ParentSelection):
    """
    Sequential selection operator.
    """
    def __init__(self):
        super().__init__()

    def select(self, opt: "Optimizer", pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, n_pair: int, n_parents: int, rng=np.random.default_rng()) -> np.ndarray:
        n_pop = len(pop_x)
        selected_idx = np.zeros((n_pair, n_parents), dtype=int)
        i_grid, j_grid = np.meshgrid(np.arange(n_pair), np.arange(n_parents), indexing='ij')
        selected_idx = i_grid * n_parents + j_grid
        return selected_idx


class SurvivorSelection:
    """
    Base class for survivor selection operators.
    """
    def __init__(self):
        pass

    def select(self, opt: "Optimizer", pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, off_x: np.ndarray, off_f: np.ndarray, off_cv: np.ndarray, n_survivors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pool_x, pool_f, pool_cv = self._create_pool(pop_x, pop_f, pop_cv, off_x, off_f, off_cv)
        survivor_idx = self._select_from_pool(opt, pool_x, pool_f, pool_cv, n_survivors)
        return pool_x[survivor_idx], pool_f[survivor_idx], pool_cv[survivor_idx]

    def _create_pool(self, pop_x: np.ndarray, pop_f: np.ndarray, pop_cv: np.ndarray, off_x: np.ndarray, off_f: np.ndarray, off_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # default: (μ + λ) selection
        # can be overridden in subclasses
        pool_x = np.vstack((pop_x, off_x))
        pool_f = np.hstack((pop_f, off_f))
        pool_cv = np.hstack((pop_cv, off_cv))
        return pool_x, pool_f, pool_cv
    
    def _select_from_pool(self, opt: "Optimizer", pool_x: np.ndarray, pool_f: np.ndarray, pool_cv: np.ndarray, n_survivors: int) -> np.ndarray:
        pass


class TruncationSelection(SurvivorSelection):
    """
    Truncation selection operator.
    """
    def __init__(self):
        super().__init__()

    def _select_from_pool(self, opt: "Optimizer", pool_x: np.ndarray, pool_f: np.ndarray, pool_cv: np.ndarray, n_survivors: int) -> np.ndarray:
        cmp = opt.problem.comparator
        cand_idx = cmp.sort(pool_f, pool_cv)
        survivor_idx = cand_idx[:n_survivors]
        return survivor_idx


class Comparator:
    """
    Base class for comparator.
    """
    def __init__(self, weights: np.ndarray, eps: float):
        self.weights = weights
        self.eps = eps

    def compare(self, fitness_a: np.ndarray, cv_a: float, fitness_b: np.ndarray, cv_b: float) -> int:
        pass

    def sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
        pass


class SingleObjectiveComparator(Comparator):
    """
    Comparator for single-objective optimization.
    """
    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        super().__init__(np.array([weight]), eps)

    def compare(self, fitness_a: np.ndarray, cv_a: float, fitness_b: np.ndarray, cv_b: float) -> int:
        if cv_a > self.eps and cv_b > self.eps:
            if cv_a < cv_b:
                return -1
            elif cv_a > cv_b:
                return 1
            else:
                return 0
        elif cv_a > self.eps and cv_b <= self.eps:
            return 1
        elif cv_a <= self.eps and cv_b > self.eps:
            return -1
        else:
            if fitness_a[0] < fitness_b[0] - self.eps:
                return -1
            elif fitness_a[0] > fitness_b[0] + self.eps:
                return 1
            else:
                return 0
    
    def sort(self, fitness: np.ndarray, cv: np.ndarray) -> np.ndarray:
        cv_key = np.where(cv > self.eps, cv, 0)
        obj_key = fitness.flatten() * self.weights[0]
        return np.lexsort((-obj_key, cv_key))


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


def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, sigma=2.0):
    # return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))
    sq_dist = scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean')
    return np.exp(-sq_dist / (2 * (sigma ** 2)))


class RBFsurrogate(Surrogate):
    def __init__(self, kernel, dim):
        self.dim = dim
        self.train_x = None
        self.train_y = None
        self.kernel = kernel
        self.weights = None
        self.kernel_matrix = None

    def fit(self, train_x, train_y):
        self.train_x = np.asarray(train_x)
        self.train_y = np.asarray(train_y)
        n_samples = len(train_x)
        self.kernel_matrix = self.kernel(self.train_x, self.train_x)
        rcond = 1 / np.linalg.cond(self.kernel_matrix)
        if rcond < np.finfo(self.kernel_matrix.dtype).eps:
            logging.warning(f"Kernel matrix is ill-conditioned. RCOND: {rcond}")
        try:
            self.weights = np.linalg.solve(self.kernel_matrix, (train_y - np.mean(train_y)))
        except np.linalg.LinAlgError:
            logging.error("Failed to solve linear system (Kernel matrix might be singular).")
            self.weights = np.zeros(n_samples)

    def predict(self, test_x):
        n_samples = len(self.train_x)
        kernel_vec = self.kernel(self.train_x, test_x).flatten()
        prediction = np.dot(kernel_vec, self.weights)
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
        self.candidate = None
        self.candidate_fit =None
        self.surrogate_model = None

        # parameters (optional)
        self.n_train = 50
        self.rsm = 0.1

    def run(self, optimizer, candidate):
        self.candidate = candidate
        n_cand = len(self.candidate)
        psm = int(self.rsm * n_cand)
        self.surrogate_model = optimizer.surrogate
        cmp = optimizer.problem.comparator

        self.candidate_fit = np.zeros(n_cand)

        # predict all candidates using surrogate model
        for i in range(n_cand):
            # get training data for candidate[i]
            train_x, train_y = optimizer.archive.get_knn(self.candidate[i], k=self.n_train)
            # train RBF model
            self.surrogate_model.fit(train_x, train_y)
            # predict candidate[i]
            self.candidate_fit[i] = self.surrogate_model.predict(self.candidate[i].reshape(1, -1))
            optimizer.dispatch(CallbackEvent.POST_SURROGATE_FIT, train_x=train_x, train_y=train_y)

        # psm individuals are evaluated using the true function
        # TODO: use cv if constraints are defined
        cand_idx = cmp.sort(self.candidate_fit, np.zeros_like(self.candidate_fit))
        self.candidate = self.candidate[cand_idx]
        self.candidate_fit = self.candidate_fit[cand_idx]

        self.candidate_eval = self.candidate[:psm]
        self.candidate_eval_fit = np.array([optimizer.problem.evaluate(ind) for ind in self.candidate_eval])
        self.candidate_fit[:psm] = self.candidate_eval_fit
        optimizer.fe += psm

        # add evaluated individuals to the archive
        for i in range(psm):
            optimizer.archive.add(self.candidate_eval[i], self.candidate_eval_fit[i])

        return self.candidate, self.candidate_fit


class CallbackEvent(Enum):
    # Optimizer.run events
    RUN_START = auto()
    RUN_END = auto()
    GENERATION_START = auto()
    GENERATION_END = auto()
    SURROGATE_START = auto()
    SURROGATE_END = auto()
    # Algorithm.ask events
    POST_CROSSOVER = auto()
    POST_MUTATION = auto()
    # ModelManager.run events (commented out for future use)
    POST_SURROGATE_FIT = auto()
    # POST_SURROGATE_PREDICT = auto()


class CallbackManager:
    def __init__(self):
        self.handlers = defaultdict(list)

    def register(self, event: CallbackEvent, func: callable):
        self.handlers[event].append(func)

    def dispatch(self, event: CallbackEvent, **kwargs):
        for handler in self.handlers[event]:
            handler(**kwargs)


def repair_clipping(**kwargs):
    problem = kwargs.get("optimizer", None).problem
    data = kwargs.get("data", None)
    repaired = np.clip(data, problem.lb, problem.ub)
    kwargs["data"] = repaired

def logging_generation(**kwargs):
    optimizer = kwargs.get("optimizer", None)
    logging.info(f"Generation {optimizer.gen} started. fe: {optimizer.fe}. Best f: {optimizer.population.get('f')[0]}")


class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, problem: Problem):
        # components
        self.problem = problem
        self.algorithm = None
        self.surrogate = None
        self.modelmanager = None
        self.termination = None
        # Archive init parameters
        self.archive_atol = 0.0
        self.archive_rtol = 0.0
        self.archive = None
        self.archive_init_size = 50
        # random setup
        self.seed = 0
        self.rng = np.random.default_rng(seed=self.seed)
        # state variables
        self.fe = 0
        self.gen = 0
        # EA parameters
        self.popsize = 40
        # callback event manager
        self.cbmanager = CallbackManager()

    def set_algorithm(self, algorithm: Algorithm):
        self.algorithm = algorithm
        return self
    
    def set_surrogate(self, surrogate: Surrogate):
        self.surrogate = surrogate
        return self
    
    def set_modelmanager(self, modelmanager: ModelManager):
        self.modelmanager = modelmanager
        return self

    def set_termination(self, termination: Termination):
        self.termination = termination
        return self

    def set_archive_init_size(self, size: int):
        self.archive_init_size = size
        return self
    
    def set_archive_atol(self, atol: float):
        self.archive_atol = atol
        return self
    
    def set_archive_rtol(self, rtol: float):
        self.archive_rtol = rtol
        return self
    
    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        return self

    def set_popsize(self, popsize: int):
        self.popsize = popsize
        return self

    def _initialize(self, n_init_archive: int):
        archive_x = self.rng.uniform(self.problem.lb, self.problem.ub, (n_init_archive, self.problem.dim))
        archive_y = np.array([self.problem.evaluate(ind) for ind in archive_x])
        # TODO: use cv if constraints are defined
        archive_sort_idx = self.problem.comparator.sort(archive_y, np.zeros_like(archive_y))
        archive_x = archive_x[archive_sort_idx]
        archive_y = archive_y[archive_sort_idx]
        self.archive = Archive.new(archive_x, archive_y, atol=self.archive_atol, rtol=self.archive_rtol)

        self.population = Population.new("x", self.archive.get("x")[:self.popsize])
        self.population.set("f", self.archive.get("y")[:self.popsize])

        self.fe = self.archive_init_size
        self.gen = 0

        self.cbmanager.register(CallbackEvent.GENERATION_START, logging_generation)
        self.cbmanager.register(CallbackEvent.POST_CROSSOVER, repair_clipping)
        self.cbmanager.register(CallbackEvent.POST_MUTATION, repair_clipping)
    
    def dispatch(self, event: CallbackEvent, **kwargs):
        kwargs["optimizer"] = self
        self.cbmanager.dispatch(event, **kwargs)

    def run(self):
        self._initialize(self.archive_init_size)
        self.dispatch(CallbackEvent.RUN_START)

        while not self.termination.is_terminated(fe=self.fe):

            self.gen += 1

            self.dispatch(CallbackEvent.GENERATION_START)

            # ask
            cand = self.algorithm.ask(self)

            self.dispatch(CallbackEvent.SURROGATE_START)

            # surrogate
            cand, cand_fit = self.modelmanager.run(self, cand)

            self.dispatch(CallbackEvent.SURROGATE_END)

            # tell
            self.algorithm.tell(self, cand, cand_fit)

            self.dispatch(CallbackEvent.GENERATION_END)

        self.dispatch(CallbackEvent.RUN_END)
