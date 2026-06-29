"""Benchmark problems for saealib.

Multi-objective suites
----------------------
- :mod:`saealib.benchmarks.zdt`  -- ZDT1-4, ZDT6 (Zitzler et al. 2000)
- :mod:`saealib.benchmarks.dtlz` -- DTLZ1-4, DTLZ7 (Deb et al. 2005)

Single-objective functions
--------------------------
- :mod:`saealib.benchmarks.singleobj` — Sphere, Rastrigin, Rosenbrock, Ackley
"""

from saealib.benchmarks.zdt import zdt1, zdt2, zdt3, zdt4, zdt5, zdt6

__all__ = [
    "zdt1",
    "zdt2",
    "zdt3",
    "zdt4",
    "zdt5",
    "zdt6",
]
