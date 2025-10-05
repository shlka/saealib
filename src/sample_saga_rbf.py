import numpy as np
import logging
from opfunu.cec_based import cec2015

from saealib.core import Optimizer, Problem, Termination, GA, MutationUniform, CrossoverBLXAlpha, RBFsurrogate, gaussian_kernel, IndividualBasedStrategy, Callback

logging.basicConfig(level=logging.INFO)


class LoggingCallback(Callback):
    def __init__(self):
        self.start_fe = 0

    def cb_run_start(self, optimizer: Optimizer):
        self.start_fe = optimizer.fe
        logging.info(f"Run started. Initial fe: {self.start_fe}")

    def cb_generation_start(self, optimizer: Optimizer):
        logging.info(f"Generation {optimizer.gen} started.")

    def cb_surrogate_start(self, optimizer: Optimizer):
        logging.info(f"Surrogate model evaluation started at fe: {optimizer.fe}")

    def cb_surrogate_end(self, optimizer: Optimizer):
        logging.info(f"Surrogate model evaluation ended at fe: {optimizer.fe}")

    def cb_generation_end(self, optimizer: Optimizer):
        best_f = optimizer.population.get("f")[0]
        logging.info(f"Generation {optimizer.gen} ended. Best f: {best_f}, fe: {optimizer.fe}")

    def cb_run_end(self, optimizer: Optimizer):
        total_fe = optimizer.fe - self.start_fe
        best_f = optimizer.population.get("f")[0]
        logging.info(f"Run ended. Total fe: {total_fe}, Best f: {best_f}")


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
    opt.surrogate = RBFsurrogate(gaussian_kernel, dim)
    opt.modelmanager = IndividualBasedStrategy()
    opt.modelmanager.knn = knn
    opt.modelmanager.rsm = rsm
    opt.callbacks = [LoggingCallback()]
    opt.run()


if __name__ == "__main__":
    main()
