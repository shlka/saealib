from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .population import Population


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
