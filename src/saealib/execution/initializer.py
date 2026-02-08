"""
Initializer module.

Initialize Population and Archive, and use them to generate an OptimizationContext.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats

from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.population import Archive, Population, PopulationAttribute
from saealib.problem import Problem


class Initializer(ABC):
    """
    Initializer class.

    Initialize Population and Archive, and use them to generate an OptimizationContext.
    """

    def _create_attrs(
        self, problem: Problem, provider: ComponentProvider
    ) -> list[PopulationAttribute]:
        """
        Create attributes for Population and Archive.

        Parameters
        ----------
        problem : Problem
            The problem instance.
        provider : ComponentProvider
            The component provider instance.

        Returns
        -------
        list[PopulationAttribute]
            The attributes for Population and Archive.
        """
        # TODO: modify it to account for the fact that there are n_obj instances of f.
        # default attributes
        attrs = [
            PopulationAttribute("x", float, (problem.dim,), default=np.nan),
            # PopulationAttribute("f",  float, (self.problem.n_obj, ), default=np.nan)
            PopulationAttribute("f", float, (), default=np.nan),
            PopulationAttribute("g", float, (problem.n_constraint,), default=0.0),
            PopulationAttribute("cv", float, (), default=0.0),
        ]
        # Retrieve attributes and classes according to the algorithm
        if provider.algorithm is not None:
            attrs_required = provider.algorithm.get_required_attrs(problem)
            ex_names = {attr.name for attr in attrs}
            for attr in attrs_required:
                if attr.name not in ex_names:
                    attrs.append(attr)
        return attrs

    def _create_context(
        self,
        problem: Problem,
        archive: Archive,
        population: Population,
        rng: np.random.Generator,
    ) -> OptimizationContext:
        """
        Create an OptimizationContext.

        Parameters
        ----------
        problem : Problem
            The problem instance.
        archive : Archive
            The archive instance.
        population : Population
            The population instance.
        rng : np.random.Generator
            The random number generator.

        Returns
        -------
        OptimizationContext
            The optimization context.
        """
        return OptimizationContext(
            problem=problem,
            population=population,
            archive=archive,
            rng=rng,
            fe=len(archive),
            gen=0,
        )

    @abstractmethod
    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationContext:
        """
        Initialize Population and Archive.

        Use them to generate an OptimizationContext.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationContext
            The optimization context.
        """
        pass


class LHSInitializer(Initializer):
    """
    Latin Hypercube Sampling Initializer.

    Attributes
    ----------
    n_init_archive : int
        The number of individuals to initialize in the archive.
    n_init_population : int
        The number of individuals to initialize in the population.
    seed : int | None
        The seed for the random number generator.
    """

    def __init__(
        self, n_init_archive: int, n_init_population: int, seed: int | None = None
    ):
        self.n_init_archive = n_init_archive
        self.n_init_population = n_init_population
        self.seed = seed

    def initialize(
        self, provider: ComponentProvider, problem: Problem
    ) -> OptimizationContext:
        """
        Initialize Population and Archive.

        Use them to generate an OptimizationContext.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationContext
            The optimization context.
        """
        rng = np.random.default_rng(self.seed)
        attrs = self._create_attrs(problem, provider)

        population = provider.algorithm.population_class(
            attrs=attrs, init_capacity=self.n_init_population
        )
        archive = provider.algorithm.archive_class(
            attrs=attrs, init_capacity=self.n_init_archive
        )

        # TODO: Assign different metadata per dimension.
        # To support mixed variable optimization in the future.
        # TODO: Supports initialization of CV and other attributes.
        # Additionally, it handles cases where the sort does not depend on f and cv.
        archive_x = scipy.stats.qmc.LatinHypercube(d=problem.dim, rng=rng).random(
            self.n_init_archive
        )
        archive_x = scipy.stats.qmc.scale(archive_x, problem.lb, problem.ub)
        archive_f = np.array([problem.evaluate(ind) for ind in archive_x])

        archive_sort_idx = problem.comparator.sort(archive_f, np.zeros_like(archive_f))
        archive_x = archive_x[archive_sort_idx]
        archive_f = archive_f[archive_sort_idx]

        # TODO: Modify the archive to register attributes predefined by the archive.
        for i in range(self.n_init_archive):
            archive.add({"x": archive_x[i], "f": archive_f[i]})

        population.extend(archive[: self.n_init_population])

        return self._create_context(problem, archive, population, rng)
