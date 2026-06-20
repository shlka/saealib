"""OptimizationContext: shared mutable state passed through the optimization loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from saealib.comparators import Comparator
    from saealib.population import Archive, ParetoArchive, Population
    from saealib.problem import Problem


@dataclass
class OptimizationContext:
    """
    Optimization Context class.

    Manage the state of the optimization process.

    Attributes
    ----------
    problem : Problem
        Problem instance.
    population : Population
        Population instance.
    archive : Archive
        Archive instance.
    rng : np.random.Generator
        Random number generator.
    fe : int
        Number of function evaluations.
    gen : int
        Number of generations.
    resumed : bool
        True when this context was restored from a checkpoint.
    metadata : dict
        Metadata.
    """

    problem: Problem

    population: Population
    archive: Archive
    pareto_archive: ParetoArchive
    rng: np.random.Generator

    fe: int = 0
    gen: int = 0
    resumed: bool = False

    metadata: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        """Return the dimension of the problem."""
        return self.problem.dim

    @property
    def n_obj(self) -> int:
        """Return the number of objectives."""
        return self.problem.n_obj

    @property
    def lb(self) -> np.ndarray:
        """Return the lower bounds of the problem."""
        return self.problem.lb

    @property
    def ub(self) -> np.ndarray:
        """Return the upper bounds of the problem."""
        return self.problem.ub

    @property
    def direction(self) -> np.ndarray:
        """Return the optimization direction of the problem."""
        return self.problem.direction

    @property
    def comparator(self) -> Comparator:
        """Return the comparator of the problem."""
        return self.problem.comparator

    def count_fe(self, count: int = 1) -> None:
        """
        Count function evaluations.

        Parameters
        ----------
        count : int
            Number of function evaluations to add.
        """
        self.fe += count

    def count_generation(self) -> None:
        """Count generations."""
        self.gen += 1

    # ------------------------------------------------------------------
    # Checkpoint: npz (best-effort reproducibility)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save optimization state to an npz file.

        Saves archive, population, Pareto archive arrays and the RNG state.
        Reproducibility is best-effort: bit-exact resume is expected within
        the same NumPy version and environment, but not guaranteed across
        versions.

        Parameters
        ----------
        path : str or Path
            Destination file path.  The ``.npz`` extension is added if absent.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        save_dict: dict[str, np.ndarray] = {}

        # Schema — serialise attribute definitions from the archive
        schema = []
        for name, attr in self.archive.schema.items():
            arr = self.archive._data[name]
            default = attr.default
            default_json: object = (
                "__nan__"
                if (isinstance(default, float) and np.isnan(default))
                else default
            )
            schema.append(
                {
                    "name": name,
                    "dtype": arr.dtype.str,
                    "shape": list(arr.shape[1:]),
                    "default": default_json,
                }
            )
        save_dict["_schema"] = np.frombuffer(
            json.dumps(schema).encode(), dtype=np.uint8
        )

        # RNG state (full bit-generator state for exact restoration)
        save_dict["_rng_state"] = np.frombuffer(
            json.dumps(self.rng.bit_generator.state).encode(), dtype=np.uint8
        )

        # Scalar counters
        save_dict["_fe"] = np.array(self.fe)
        save_dict["_gen"] = np.array(self.gen)

        # Archive arrays
        n_arch = len(self.archive)
        save_dict["_archive_size"] = np.array(n_arch)
        for name, arr in self.archive._data.items():
            save_dict[f"archive__{name}"] = arr[:n_arch]

        # Population arrays
        n_pop = len(self.population)
        save_dict["_pop_size"] = np.array(n_pop)
        for name, arr in self.population._data.items():
            save_dict[f"pop__{name}"] = arr[:n_pop]

        # Pareto archive arrays
        n_pareto = len(self.pareto_archive)
        save_dict["_pareto_size"] = np.array(n_pareto)
        for name, arr in self.pareto_archive._data.items():
            save_dict[f"pareto__{name}"] = arr[:n_pareto]

        np.savez(p, **save_dict)  # type: ignore  # np.savez **kwargs typed as ArrayLike; save_dict is dict[str, Any]

    @classmethod
    def load(cls, path: str | Path, problem: Problem) -> OptimizationContext:
        """
        Restore an OptimizationContext from an npz checkpoint file.

        The returned context has ``resumed=True``.

        Parameters
        ----------
        path : str or Path
            Path to the npz file.  The ``.npz`` extension is added if absent.
        problem : Problem
            The problem instance to attach (must match the one used when saving).

        Returns
        -------
        OptimizationContext
        """
        from saealib.population import (
            Archive,
            ParetoArchive,
            Population,
            PopulationAttribute,
        )

        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        data = np.load(p, allow_pickle=False)

        # Reconstruct attribute schema
        schema_list = json.loads(bytes(data["_schema"]).decode())
        attrs = []
        for s in schema_list:
            default = s["default"]
            if default == "__nan__":
                default = np.nan
            attrs.append(
                PopulationAttribute(
                    name=s["name"],
                    dtype=np.dtype(s["dtype"]),
                    shape=tuple(s["shape"]),
                    default=default,
                )
            )

        n_arch = int(data["_archive_size"])
        n_pop = int(data["_pop_size"])
        n_pareto = int(data["_pareto_size"])
        attr_names = [a.name for a in attrs]

        archive = Archive(attrs=attrs, init_capacity=max(n_arch, 1))
        if n_arch > 0:
            archive.extend({name: data[f"archive__{name}"] for name in attr_names})

        population = Population(attrs=attrs, init_capacity=max(n_pop, 1))
        if n_pop > 0:
            population.extend({name: data[f"pop__{name}"] for name in attr_names})

        pareto_archive = ParetoArchive(
            attrs=attrs,
            init_capacity=max(n_pareto, 1),
            direction=problem.direction,
        )
        if n_pareto > 0:
            pareto_archive.extend(
                {name: data[f"pareto__{name}"] for name in attr_names}
            )

        # Restore RNG to exact saved state
        rng_state = json.loads(bytes(data["_rng_state"]).decode())
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state

        return cls(
            problem=problem,
            population=population,
            archive=archive,
            pareto_archive=pareto_archive,
            rng=rng,
            fe=int(data["_fe"]),
            gen=int(data["_gen"]),
            resumed=True,
        )
