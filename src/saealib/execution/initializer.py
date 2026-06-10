"""Initializer: builds the initial Population, Archive, and OptimizationContext."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.stats

from saealib.context import OptimizationContext
from saealib.optimizer import ComponentProvider
from saealib.population import Archive, ParetoArchive, Population, PopulationAttribute
from saealib.problem import Problem


class Initializer(ABC):
    """Abstract base for classes that set up the initial optimization context."""

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
        attrs = [
            PopulationAttribute("x", float, (problem.dim,), default=np.nan),
            PopulationAttribute("f", float, (problem.n_obj,), default=np.nan),
            PopulationAttribute("g", float, (problem.n_constraints,), default=0.0),
            PopulationAttribute("cv", float, (), default=0.0),
        ]
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
        pareto_archive: ParetoArchive,
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
        pareto_archive : ParetoArchive
            The Pareto archive instance.
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
            pareto_archive=pareto_archive,
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
        Initialize Population and Archive with LHS samples.

        Parameters
        ----------
        provider : ComponentProvider
            The component provider instance.
        problem : Problem
            The problem instance.

        Returns
        -------
        OptimizationContext
        """
        rng = np.random.default_rng(self.seed)
        attrs = self._create_attrs(problem, provider)

        population = provider.algorithm.population_class(
            attrs=attrs, init_capacity=self.n_init_population
        )
        archive = provider.algorithm.archive_class(
            attrs=attrs, init_capacity=self.n_init_archive
        )
        pareto_archive = provider.algorithm.create_pareto_archive(
            attrs=attrs, init_capacity=self.n_init_archive, problem=problem
        )

        # TODO: Assign different metadata per dimension (mixed variable support).
        # TODO: Support initialization of CV and other attributes.
        archive_x = scipy.stats.qmc.LatinHypercube(d=problem.dim, rng=rng).random(
            self.n_init_archive
        )
        archive_x = scipy.stats.qmc.scale(archive_x, problem.lb, problem.ub)
        result = provider.evaluator.evaluate_batch(archive_x, problem)

        # TODO: Register algorithm-specific attributes in the archive.
        for i in range(self.n_init_archive):
            data = {
                "x": archive_x[i],
                "f": result.f[i],
                "g": result.g[i],
                "cv": float(result.cv[i]),
            }
            archive.add(data)
            pareto_archive.add(data)

        sorted_idx = problem.comparator.sort_population(archive)
        archive_sorted = archive.extract(sorted_idx)
        archive.clear()
        archive.extend(archive_sorted)

        population.extend(archive[: self.n_init_population])

        return self._create_context(problem, archive, pareto_archive, population, rng)
