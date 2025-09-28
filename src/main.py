import logging

import numpy as np
from opfunu.cec_based import cec2015

logging.basicConfig(level=logging.INFO)


def main():
    # parameters
    dim = 10
    seed = 1
    knn = 5
    rsm = 0.9

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
    num_cross = 2 * round(crossover_rate * popsize / 2)
    num_mut = round(mutation_rate * popsize)

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
        parent_crossover = parent[randpop_idx[:num_cross]]
        offspring_crossover = []
        for i in range(0, num_cross, 2):
            p1 = parent_crossover[i]
            p2 = parent_crossover[i + 1]

            # TODO: implement BLX-alpha
            p1, p2 = crossover_blx_alpha(p1, p2, gamma, lb, ub)

            offspring_crossover.append(p1)
            offspring_crossover.append(p2)

        # mutation
        parent_mutation = parent[randpop_idx[num_cross:]]
        offspring_mutation = []
        for i in range(num_mut):
            p = parent_mutation[i]

            # TODO: implement uniform mutation
            p = mutation_uniform(p, mutation_rate, lb, ub)

            offspring_mutation.append(p)

        offspring = np.vstack((offspring_crossover, offspring_mutation))

        psm = int(rsm * popsize)

        # surrogate model
        for i in range(popsize):
            # TODO: implement surrogate model
            # get training data for offspring[i]
            # train RBF model
            # predict offspring[i]
            pass

        # TODO: implement selection
        # sort by predicted fitness
        # select psm individuals to evaluate with the true function
        fe = fe + psm
        # add evaluated individuals to the archive
        # select a best solution in parent
        # update population and fitness

        logging.info(f"fbest: {fit[0]}, fe: {fe}")


if __name__ == "__main__":
    main()
