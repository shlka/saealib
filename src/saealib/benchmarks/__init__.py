"""Benchmark problems for saealib.

Multi-objective suites
----------------------
- :mod:`saealib.benchmarks.zdt`      -- ZDT1-6 (Zitzler et al. 2000)
- :mod:`saealib.benchmarks.dtlz`     -- DTLZ1-7 (Deb et al. 2005)

Single-objective functions
--------------------------
- :mod:`saealib.benchmarks.singleobj` -- Sphere, Rosenbrock, Ackley, Rastrigin
"""

from saealib.benchmarks.dtlz import dtlz1, dtlz2, dtlz3, dtlz4, dtlz5, dtlz6, dtlz7
from saealib.benchmarks.singleobj import ackley, rastrigin, rosenbrock, sphere
from saealib.benchmarks.zdt import zdt1, zdt2, zdt3, zdt4, zdt5, zdt6

__all__ = [
    "ackley",
    "dtlz1",
    "dtlz2",
    "dtlz3",
    "dtlz4",
    "dtlz5",
    "dtlz6",
    "dtlz7",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt5",
    "zdt6",
]
