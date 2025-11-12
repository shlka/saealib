"""
Model manager module.

This module defines the base class `ModelManager` and the derived class `IndividualBasedStrategy`
for managing surrogate models in evolutionary optimization.
These classes provide how to evaluate candidate solutions using a combination of true evaluations and surrogate model predictions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

from saealib.callback import CallbackEvent

if TYPE_CHECKING:
    from saealib.optimizer import Optimizer



class ModelManager(ABC):
    """
    Base class for surrogate model manager.
    """
    @abstractmethod
    def run(self, optimizer: Optimizer, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the model manager.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.
        candidate : np.ndarray
            The candidate solutions to be evaluated or predicted.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the candidate solutions and their corresponding fitness values.
            (candidate_x, candidate_y)
        """
        pass


class IndividualBasedStrategy(ModelManager):
    """
    Individual-based strategy for surrogate model management.

    A method in which a certain percentage of individuals are evaluated with a true evaluation function
    for each candidate solution per generation, and the remainder are predicted by a surrogate model.

    Attributes
    ----------
    candidate : np.ndarray
        The candidate solutions. shape=(n_candidate, n_dimension)
    candidate_fit : np.ndarray
        The fitness values of the candidate solutions. shape=(n_candidate,)
    surrogate_model : Surrogate
        The surrogate model used for predictions.
    n_train : int
        Number of training samples for surrogate model.
    rsm : float
        Ratio of real evaluations.
    """
    def __init__(self):
        """
        Initialize IndividualBasedStrategy class.
        """
        self.candidate = None
        self.candidate_fit =None
        self.surrogate_model = None

        # parameters (optional) TODO: make them configurable
        self.n_train = 50
        self.rsm = 0.1

    def run(self, optimizer: Optimizer, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the individual-based strategy.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.
        candidate : np.ndarray
            The candidate solutions to be evaluated or predicted.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the candidate solutions and their corresponding fitness values.
            (candidate_x, candidate_y)
        """
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
