import numpy as np
import logging
from opfunu.cec_based import cec2015

from saealib.core import Population, Archive, Optimizer, Problem, Termination, GA, MutationUniform, CrossoverBLXAlpha

def main():
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1

    # benchmark function
    f1 = cec2015.F12015(ndim=10)

    opt = Optimizer()
    opt.problem = Problem(f1.evaluate, dim, lb=-100, ub=100)
    opt.algorithm = GA(
        crossover=CrossoverBLXAlpha(crossover_rate=0.7, gamma=0.4, lb=-100, ub=100),
        mutation=MutationUniform(mutation_rate=0.3, lb=-100, ub=100),
        selection=None  # Define a selection method here
    )
    opt.termination = Termination(fe=200 * dim)
    opt.archive_atol = 0.0
    opt.seed = seed
    opt.archive_init_size = 5 * dim
    opt.run()


if __name__ == "__main__":
    main()
