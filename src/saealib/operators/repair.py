import numpy as np


def repair_clipping(data, **kwargs):
    problem = kwargs.get("optimizer", None).problem
    repaired = np.clip(data, problem.lb, problem.ub)
    return repaired
