import numpy as np
import logging
import cProfile
from opfunu.cec_based import cec2015

from saealib.core import Optimizer, Problem, Termination, GA, MutationUniform, CrossoverBLXAlpha, RBFsurrogate, gaussian_kernel, IndividualBasedStrategy

logging.basicConfig(level=logging.INFO)


def main():
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1
    ub = [100] * dim
    lb = [-100] * dim

    # benchmark function
    f1 = cec2015.F12015(ndim=10)

    problem = Problem(f1.evaluate, dim, lb=lb, ub=ub)
    algorithm = GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, gamma=0.4, lb=lb, ub=ub),
        mutation=MutationUniform(mutation_rate=0.3, lb=lb, ub=ub),
        selection=None  # Define a selection method here
    )
    termination = Termination(fe=200 * dim)
    surrogate = RBFsurrogate(gaussian_kernel, dim)
    modelmanager = IndividualBasedStrategy()
    modelmanager.knn = knn
    modelmanager.rsm = rsm

    opt = (Optimizer(problem)
            .set_algorithm(algorithm)
            .set_termination(termination)
            .set_surrogate(surrogate)
            .set_modelmanager(modelmanager)
            .set_archive_init_size(5 * dim)
            .set_archive_atol(0.0)
            .set_seed(seed)
    )
    opt.run()


if __name__ == "__main__":
    cProfile.run("main()")
