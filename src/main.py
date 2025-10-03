import logging
import math

import numpy as np
from opfunu.cec_based import cec2015

logging.basicConfig(level=logging.INFO)


def main():
    # parameters
    dim = 10
    seed = 1
    knn = 50
    rsm = 0.1

    # benchmark function
    f1 = cec2015.F12015(ndim=10)

    ibrbf_ga(f1, dim, seed, knn, rsm)


def ibrbf_ga(func, dim, seed, knn, rsm):
    # search range
    lb = -100
    ub = 100

    # random seed
    np.random.seed(seed)

    # parameters
    maxfe = 200 * dim
    popsize = 40
    init_archive_size = 5 * dim
    crossover_rate = 0.7
    gamma = 0.4
    mutation_rate = 0.3
    mutation_rate_inner = 0.3

    # initialize archive
    archive_x = np.random.uniform(lb, ub, (init_archive_size, dim))
    archive_y = np.array([func.evaluate(ind) for ind in archive_x])
    archive_x = archive_x[np.argsort(archive_y)]
    archive_y = np.sort(archive_y)

    # initialize population
    pop = archive_x[:popsize]
    fit = archive_y[:popsize]

    fe = init_archive_size
    logging.info(f"fbest: {fit[0]}, fe: {fe}")

    while fe < maxfe:
        randpop_idx = np.random.permutation(popsize)
        parent = pop[randpop_idx]
        parent_fit = fit[randpop_idx]

        # crossover
        offspring = np.empty((0, dim))
        for i in range(0, popsize, 2):
            if np.random.rand() < crossover_rate:
                p1 = parent[i]
                p2 = parent[i + 1]
                o1, o2 = crossover_blx_alpha(p1, p2, gamma, lb, ub)
                offspring = np.vstack((offspring, o1, o2))
            else:
                offspring = np.vstack((offspring, parent[i], parent[i + 1]))

        # mutation
        for i in range(popsize):
            if np.random.rand() < mutation_rate:
                p = offspring[i]
                offspring[i] = mutation_uniform(p, mutation_rate_inner, lb, ub)

        psm = int(rsm * popsize)

        # surrogate model
        offspring_fit = np.array([])
        for i in range(popsize):
            # TODO: implement surrogate model
            # get training data for offspring[i]
            train_x, train_y = get_neighbors(archive_x, archive_y, offspring[i], knn)
            # train RBF model
            rbf_model = RBF(gaussian_kernel, dim)
            rbf_model.fit(train_x, train_y)
            # predict offspring[i]
            offspring_fit = np.append(offspring_fit, rbf_model.predict(offspring[i].reshape(1, -1)))

        # TODO: remove this
        # offspring_fit = [func.evaluate(ind) for ind in offspring]

        # TODO: implement selection
        # sort by predicted fitness
        offspring = offspring[np.argsort(offspring_fit)]
        offspring_fit = np.sort(offspring_fit)
        # select psm individuals to evaluate with the true function
        offspring_eval = offspring[:psm]
        offspring_eval_fit = np.array([func.evaluate(ind) for ind in offspring_eval])
        offspring_fit[:psm] = offspring_eval_fit
        fe = fe + psm
        # add evaluated individuals to the archive
        archive_x = np.vstack((archive_x, offspring_eval))
        archive_y = np.hstack((archive_y, offspring_eval_fit))
        # select a best solution in parent
        best_idx = np.argmin(parent_fit)
        parent_best = parent[best_idx]
        parent_best_fit = parent_fit[best_idx]
        parent = np.delete(parent, best_idx, axis=0)
        parent_fit = np.delete(parent_fit, best_idx, axis=0)
        # update population and fitness
        pop = np.vstack((parent_best, parent, offspring))
        fit = np.hstack((parent_best_fit, parent_fit, offspring_fit))
        pop = pop[np.argsort(fit)]
        fit = np.sort(fit)
        pop = pop[:popsize]
        fit = fit[:popsize]

        logging.info(f"fbest: {archive_y.min()}, fe: {fe}")


def crossover_blx_alpha(p1, p2, gamma, lb, ub):
    dim = len(p1)
    alpha = np.random.uniform(-gamma, 1 + gamma, size=dim)
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = (1 - alpha) * p1 + alpha * p2
    # TODO: implement boundary handling
    c1 = np.clip(c1, lb, ub)
    c2 = np.clip(c2, lb, ub)
    return c1, c2


def mutation_uniform(p, mutation_rate, lb, ub):
    dim = len(p)
    mutation_bool = np.random.random(dim) < mutation_rate
    c = p.copy()
    for i in range(dim):
        if np.random.rand() < mutation_rate:
            c[i] = np.random.uniform(lb, ub)
    return c

def get_neighbors(archive_x, archive_y, x, k):
    distances = np.linalg.norm(archive_x - x, axis=1)
    neighbor_idx = np.argsort(distances)[:k]
    return archive_x[neighbor_idx], archive_y[neighbor_idx]


def gaussian_kernel(x1, x2, sigma=2.0):
    return math.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

class RBF:
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



if __name__ == "__main__":
    main()
