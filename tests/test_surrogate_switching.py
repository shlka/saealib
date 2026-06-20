"""Tests for surrogate/switching.py."""

from unittest.mock import MagicMock

import pytest

from saealib.surrogate.accuracy import SurrogateAccuracy
from saealib.surrogate.switching import (
    AccuracyBasedSurrogateSwitcher,
    GenCtrlSwitcher,
    ManagerSwitcher,
    StrategySwitcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acc(spearman: float, n: int = 10) -> SurrogateAccuracy:
    return SurrogateAccuracy(metrics={"spearman": spearman}, n_samples=n)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


def test_abc_is_abstract() -> None:
    with pytest.raises(TypeError):
        AccuracyBasedSurrogateSwitcher()  # type: ignore[abstract]  # intentional: testing abstract instantiation raises TypeError


# ---------------------------------------------------------------------------
# ManagerSwitcher
# ---------------------------------------------------------------------------


class TestManagerSwitcher:
    def _make(self, threshold=0.5):
        primary, fallback = MagicMock(name="primary"), MagicMock(name="fallback")
        sw = ManagerSwitcher(primary, fallback, threshold=threshold)
        return primary, fallback, sw

    def test_returns_primary_above_threshold(self) -> None:
        primary, _, sw = self._make()
        assert sw.switch(_acc(0.8)) is primary

    def test_returns_primary_at_threshold(self) -> None:
        primary, _, sw = self._make(threshold=0.5)
        assert sw.switch(_acc(0.5)) is primary

    def test_returns_fallback_below_threshold(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(_acc(0.3)) is fallback

    def test_returns_fallback_when_accuracy_none(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(None) is fallback

    def test_returns_fallback_when_metric_missing(self) -> None:
        _, fallback, sw = self._make()
        acc = SurrogateAccuracy(metrics={}, n_samples=5)
        assert sw.switch(acc) is fallback

    def test_custom_metric(self) -> None:
        primary, fallback = MagicMock(), MagicMock()
        sw = ManagerSwitcher(primary, fallback, metric="r2", threshold=0.7)
        high = SurrogateAccuracy(metrics={"r2": 0.9}, n_samples=5)
        low = SurrogateAccuracy(metrics={"r2": 0.5}, n_samples=5)
        assert sw.switch(high) is primary
        assert sw.switch(low) is fallback

    def test_defaults(self) -> None:
        sw = ManagerSwitcher(MagicMock(), MagicMock())
        assert sw.metric == "spearman"
        assert sw.threshold == 0.5

    def test_sequence(self) -> None:
        primary, fallback, sw = self._make(threshold=0.6)
        snapshots = [None, _acc(0.4), _acc(0.7), _acc(0.55), _acc(0.9)]
        expected = [fallback, fallback, primary, fallback, primary]
        for acc, exp in zip(snapshots, expected):
            assert sw.switch(acc) is exp


# ---------------------------------------------------------------------------
# StrategySwitcher
# ---------------------------------------------------------------------------


class TestStrategySwitcher:
    def _make(self, threshold=0.56):
        primary, fallback = MagicMock(name="ps"), MagicMock(name="ib")
        sw = StrategySwitcher(primary, fallback, threshold=threshold)
        return primary, fallback, sw

    def test_returns_primary_above_threshold(self) -> None:
        primary, _, sw = self._make()
        assert sw.switch(_acc(0.8)) is primary

    def test_returns_fallback_below_threshold(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(_acc(0.3)) is fallback

    def test_returns_fallback_when_none(self) -> None:
        _, fallback, sw = self._make()
        assert sw.switch(None) is fallback

    def test_default_threshold_is_056(self) -> None:
        sw = StrategySwitcher(MagicMock(), MagicMock())
        assert sw.threshold == 0.56

    def test_default_metric_is_spearman(self) -> None:
        sw = StrategySwitcher(MagicMock(), MagicMock())
        assert sw.metric == "spearman"


# ---------------------------------------------------------------------------
# GenCtrlSwitcher
# ---------------------------------------------------------------------------


class TestGenCtrlSwitcherInit:
    def test_invalid_update_rate_zero(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(update_rate=0.0)

    def test_invalid_update_rate_above_one(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(update_rate=1.1)

    def test_invalid_gm_min_negative(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(gm_min=-1)

    def test_invalid_gm_max_less_than_min(self) -> None:
        with pytest.raises(ValueError):
            GenCtrlSwitcher(gm_max=2, gm_min=5)


class TestGenCtrlSwitcherSwitch:
    def test_perfect_accuracy_returns_gm_max(self) -> None:
        sw = GenCtrlSwitcher(gm_max=5, update_rate=1.0)
        assert sw.switch(_acc(1.0)) == 5

    def test_worst_accuracy_returns_gm_min(self) -> None:
        sw = GenCtrlSwitcher(gm_max=5, gm_min=0, update_rate=1.0)
        assert sw.switch(_acc(-1.0)) == 0

    def test_medium_accuracy_intermediate(self) -> None:
        # spearman=0.0 -> eps=0.5 -> quality=0.5 -> gm=5
        sw = GenCtrlSwitcher(gm_max=10, update_rate=1.0)
        assert sw.switch(_acc(0.0)) == 5

    def test_none_accuracy_does_not_change_state(self) -> None:
        sw = GenCtrlSwitcher(gm_max=10, update_rate=1.0, initial_error=0.5)
        before = sw.smoothed_error
        sw.switch(None)
        assert sw.smoothed_error == before

    def test_smoothing_converges_to_gm_max(self) -> None:
        sw = GenCtrlSwitcher(gm_max=10, update_rate=0.5)
        for _ in range(30):
            result = sw.switch(_acc(1.0))
        assert result == 10

    def test_gm_min_clamp(self) -> None:
        sw = GenCtrlSwitcher(gm_max=5, gm_min=2, update_rate=1.0)
        assert sw.switch(_acc(-1.0)) >= 2

    def test_gm_max_clamp(self) -> None:
        sw = GenCtrlSwitcher(gm_max=3, update_rate=1.0)
        assert sw.switch(_acc(1.0)) <= 3

    def test_smoothed_error_is_public(self) -> None:
        sw = GenCtrlSwitcher(initial_error=0.3)
        assert sw.smoothed_error == 0.3
