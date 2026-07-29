"""Internal helper for batch-API dispatch consistency (Issue #224 fix).

Used by GA (crossover_batch/mutate_batch dispatch) and ParetoMixin
(dominates_many dispatch) to decide whether an object's batch method
reflects its scalar method(s), accounting for subclassing.
"""

from __future__ import annotations


def batch_override_is_consistent(
    obj: object, batch_method: str, *scalar_methods: str
) -> bool:
    """
    Return True iff *obj*'s batch method is at least as derived as its scalar methods.

    A batch/scalar method pair is "consistent" when whichever class in
    ``type(obj).__mro__`` most recently overrides ``batch_method`` is the
    same class (or a more-derived class) than whichever class overrides
    each of ``scalar_methods``. If a subclass overrides a scalar method
    without also overriding the batch method, the batch method comes from
    a less-derived ancestor and this returns False -- signaling that the
    batch path would silently bypass the subclass's scalar override and
    should not be used.

    Parameters
    ----------
    obj : object
        The instance to inspect.
    batch_method : str
        Name of the (optional, opt-in) batch method, e.g. "crossover_batch".
    *scalar_methods : str
        Name(s) of the scalar method(s) the batch method must stay
        consistent with, e.g. "crossover", or ("dominance_matrix", "dominates").

    Returns
    -------
    bool
        True if it's safe to dispatch to ``batch_method``; False if the
        caller should fall back to the scalar path.
    """
    mro = type(obj).__mro__

    def _definer_index(name: str) -> int:
        for i, klass in enumerate(mro):
            if name in klass.__dict__:
                return i
        raise AttributeError(f"{name!r} not found in the MRO of {type(obj).__name__}")

    batch_idx = _definer_index(batch_method)
    return all(batch_idx <= _definer_index(name) for name in scalar_methods)
