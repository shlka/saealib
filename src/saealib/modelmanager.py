"""
Model manager module.

This module defines the base class `ModelManager` and
the derived class `IndividualBasedStrategy` for managing
surrogate models in evolutionary optimization.
These classes provide how to evaluate candidate solutions using a combination
of true evaluations and surrogate model predictions.
This module defines the base class `ModelManager` and
the derived class `IndividualBasedStrategy` for managing
surrogate models in evolutionary optimization.
These classes provide how to evaluate candidate solutions using a combination
of true evaluations and surrogate model predictions.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing import TYPE_CHECKING

import numpy as np

from saealib.callback import CallbackEvent

if TYPE_CHECKING:
    from saealib.context import OptimizationContext
    from saealib.optimizer import ComponentProvider, Optimizer

    from saealib.optimizer import Optimizer, OptimizationContext, ComponentProvider


class ModelManager(ABC):
    """Base class for surrogate model manager."""

    """Base class for surrogate model manager."""

    @abstractmethod
    def run(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        candidate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the model manager.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        candidate : np.ndarray
            The candidate solutions to be evaluated or predicted.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the candidate solutions
            and their corresponding fitness values.
            A tuple containing the candidate solutions
            and their corresponding fitness values.
            (candidate_x, candidate_y)
        """
        pass

    def step(self, optimizer: Optimizer) -> None:
        """
        Step function called at each generation.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.
        """
        pass


class IndividualBasedStrategy(ModelManager):
    """
    Individual-based strategy for surrogate model management.

    A method in which a certain percentage of individuals are evaluated
    with a true evaluation function for each candidate solution per generation,
    and the remainder are predicted by a surrogate model.
    A method in which a certain percentage of individuals are evaluated
    with a true evaluation function for each candidate solution per generation,
    and the remainder are predicted by a surrogate model.

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
        """Initialize IndividualBasedStrategy class."""
        """Initialize IndividualBasedStrategy class."""
        self.candidate = None
        self.candidate_fit = None
        self.candidate_fit = None
        self.surrogate_model = None

        # parameters (optional) TODO: make them configurable
        self.n_train = 50
        self.rsm = 0.1

    def run(
        self,
        ctx: OptimizationContext,
        provider: ComponentProvider,
        candidate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the individual-based strategy.

        Parameters
        ----------
        ctx : OptimizationContext
            A dataclass object that holds internal information about the Optimizer.
        provider : ComponentProvider
            Objects of the class in which the component is exposed (ex. Optimizer).
        candidate : np.ndarray
            The candidate solutions to be evaluated or predicted.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the candidate solutions
            and their corresponding fitness values.
            A tuple containing the candidate solutions
            and their corresponding fitness values.
            (candidate_x, candidate_y)
        """
        self.candidate = candidate
        n_cand = len(self.candidate)
        psm = int(self.rsm * n_cand)
        self.surrogate_model = provider.surrogate
        cmp = ctx.comparator

        self.candidate_fit = np.zeros(n_cand)

        # predict all candidates using surrogate model
        for i in range(n_cand):
            # get training data for candidate[i]
            train_idx, _ = ctx.archive.get_knn(self.candidate[i], k=self.n_train)
            train_x = ctx.archive.get_array("x")[train_idx]
            train_f = ctx.archive.get_array("f")[train_idx]
            # train RBF model
            self.surrogate_model.fit(train_x, train_f)
            # predict candidate[i]
            self.candidate_fit[i] = self.surrogate_model.predict(self.candidate[i])
            provider.dispatch(
                CallbackEvent.POST_SURROGATE_FIT,
                None,
                ctx=ctx,
                train_x=train_x,
                train_f=train_f,
                center=self.candidate[i],
            )

        # psm individuals are evaluated using the true function
        # TODO: use cv if constraints are defined
        cand_idx = cmp.sort(self.candidate_fit, np.zeros_like(self.candidate_fit))
        self.candidate = self.candidate[cand_idx]
        self.candidate_fit = self.candidate_fit[cand_idx]

        self.candidate_eval = self.candidate[:psm]
        self.candidate_eval_fit = np.array(
            [ctx.problem.evaluate(ind) for ind in self.candidate_eval]
        )
        self.candidate_eval_fit = np.array(
            [ctx.problem.evaluate(ind) for ind in self.candidate_eval]
        )
        self.candidate_fit[:psm] = self.candidate_eval_fit
        ctx.count_fe(psm)

        # add evaluated individuals to the archive
        for i in range(psm):
            ctx.archive.add(
                {"x": self.candidate_eval[i], "f": self.candidate_eval_fit[i]}
            )
            ctx.archive.add(
                {"x": self.candidate_eval[i], "f": self.candidate_eval_fit[i]}
            )

        return self.candidate, self.candidate_fit

    def _predict_all(
        self, optimizer: Optimizer, candidate: np.ndarray, n_train: int
    ) -> np.ndarray:
        """
        Predict fitness values for all candidate solutions using the surrogate model.


        using k-NN training data selection.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.
        candidate : np.ndarray
            The candidate solutions to be predicted.

        Returns
        -------
        np.ndarray
            The predicted fitness values for the candidate solutions.
        """
        n_cand = len(candidate)
        candidate_fit = np.zeros(n_cand)

        for i in range(n_cand):
            # get training data for candidate[i]
            train_idx, _ = optimizer.archive.get_knn(self.candidate[i], k=self.n_train)
            train_x = optimizer.archive.get_array("x")[train_idx]
            train_f = optimizer.archive.get_array("f")[train_idx]
            # train RBF model
            self.surrogate_model.fit(train_x, train_f)
            # predict candidate[i]
            candidate_fit[i] = self.surrogate_model.predict(candidate[i])
            optimizer.dispatch(
                CallbackEvent.POST_SURROGATE_FIT,
                None,
                train_x=train_x,
                train_y=train_f,
                center=candidate[i],
            )
            optimizer.dispatch(
                CallbackEvent.POST_SURROGATE_FIT,
                None,
                train_x=train_x,
                train_f=train_f,
                center=candidate[i],
            )

        return candidate_fit

    def _eval_true(
        self, optimizer: Optimizer, candidate: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate candidate solutions using the true evaluation function.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.
        candidate : np.ndarray
            The candidate solutions to be evaluated.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the candidate solutions
            and their corresponding fitness values.
            A tuple containing the candidate solutions
            and their corresponding fitness values.
            (candidate_x, candidate_y)
        """
        n_cand = len(candidate)
        candidate_y = np.array([optimizer.problem.evaluate(ind) for ind in candidate])
        optimizer.fe += n_cand

        # add evaluated individuals to the archive
        for i in range(n_cand):
            optimizer.archive.add(candidate[i], candidate_y[i])

        return candidate, candidate_y


    def step(self, optimizer: Optimizer) -> None:
        """
        Step function called at each generation.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer instance.
        """
        cand = optimizer.algorithm.ask(optimizer)
        n_cand = len(cand)
        n_eval = int(self.rsm * n_cand)
        if n_eval < 1:
            n_eval = 1

        optimizer.dispatch(CallbackEvent.SURROGATE_START)

        cand_x, cand_y = self.run(optimizer, cand)

        optimizer.dispatch(
            CallbackEvent.SURROGATE_END, None, candidate_x=cand_x, candidate_y=cand_y
        )
        optimizer.dispatch(
            CallbackEvent.SURROGATE_END, None, candidate_x=cand_x, candidate_y=cand_y
        )

        optimizer.algorithm.tell(optimizer, cand_x, cand_y)
