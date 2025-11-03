from typing import TYPE_CHECKING

import numpy as np

from saealib.callback import CallbackEvent

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer



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

    def run(self, optimizer: Optimizer, candidate: np.ndarray):
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
            self.candidate_fit[i] = self.surrogate_model.predict(self.candidate[i])
            optimizer.dispatch(CallbackEvent.POST_SURROGATE_FIT, None, train_x=train_x, train_y=train_y, center=self.candidate[i])

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
