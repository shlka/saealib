"""Asynchronous evaluation scheduling."""

from __future__ import annotations

import time
from collections.abc import Iterable
from math import fsum
from typing import TYPE_CHECKING, Any

import numpy as np

from saealib.callback import PostEvaluationEvent
from saealib.exceptions import (
    CheckpointError,
    EvaluationFatalError,
    EvaluationProtocolError,
    EvaluationSubmissionError,
    ValidationError,
)
from saealib.execution.evaluator import (
    EvaluationErrorInfo,
    EvaluationRequest,
    EvaluationStatus,
    EvaluationUpdate,
    Evaluator,
    PendingEvaluation,
)
from saealib.policies.feedback import FeedbackPolicy, FeedbackResult

if TYPE_CHECKING:
    from saealib.context import OptimizationState


class AsyncScheduler:
    """Coordinate asynchronous requests and serialized state mutation."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        max_pending: int = 1,
        max_reserved_fe: int | None = None,
        max_reserved_cost: float | None = None,
        timeout: float | None = None,
        retry_limit: int = 0,
        feedback_policy: FeedbackPolicy | None = None,
        algorithm: Any = None,
        callback_manager: Any = None,
    ) -> None:
        if max_pending < 1 or retry_limit < 0:
            raise ValidationError("max_pending and retry_limit are invalid")
        if timeout is not None and timeout < 0:
            raise ValidationError("timeout must be non-negative")
        if max_reserved_fe is not None and max_reserved_fe < 0:
            raise ValidationError("max_reserved_fe must be non-negative")
        if max_reserved_cost is not None and max_reserved_cost < 0:
            raise ValidationError("max_reserved_cost must be non-negative")
        if timeout is not None and (
            type(evaluator).cancel is Evaluator.cancel
            and type(evaluator).detach is Evaluator.detach
        ):
            raise ValidationError(
                "timeout requires an evaluator with cancellation or detach support"
            )
        self.evaluator = evaluator
        self.max_pending = max_pending
        self.max_reserved_fe = max_reserved_fe
        self.max_reserved_cost = max_reserved_cost
        self.timeout = timeout
        self.retry_limit = retry_limit
        self.feedback_policy = feedback_policy
        self.algorithm = algorithm
        self.callback_manager = callback_manager
        self._fatal_states: dict[int, OptimizationState] = {}

    def pending_candidate_ids(self, state: OptimizationState) -> np.ndarray:
        """Return candidate IDs reserved by pending requests."""
        values = [
            pending.request.candidate_ids
            for pending in state.pending_evaluations.values()
        ]
        return (
            np.unique(np.concatenate(values)).astype(np.int64, copy=False)
            if values
            else np.empty(0, dtype=np.int64)
        )

    def reserved_fe(self, state: OptimizationState) -> int:
        """Return reserved candidate count."""
        return sum(
            len(pending.request.candidate_ids)
            for pending in state.pending_evaluations.values()
        )

    def reserved_cost(self, state: OptimizationState) -> float:
        """Return reserved estimated cost."""
        return fsum(
            pending.reserved_cost for pending in state.pending_evaluations.values()
        )

    def submit(
        self, state: OptimizationState, requests: Iterable[EvaluationRequest]
    ) -> OptimizationState:
        """Submit disjoint requests within capacity and budget limits."""
        requests = tuple(requests)
        if len(requests) > 1 and not self.evaluator.supports_batch_rollback():
            raise EvaluationProtocolError(
                "batch submission requires rollback-capable evaluator"
            )
        if len(state.pending_evaluations) + len(requests) > self.max_pending:
            raise EvaluationProtocolError("worker capacity would be exceeded")
        if state.offspring is None:
            raise EvaluationProtocolError("asynchronous submission requires offspring")
        existing = set(map(int, self.pending_candidate_ids(state)))
        reserved_fe = self.reserved_fe(state)
        reserved_cost = self.reserved_cost(state)
        seen_requests: set[int] = set()
        plans: list[tuple[EvaluationRequest, float]] = []
        for request in requests:
            request_id = int(request.request_id)
            if request_id in seen_requests or request_id in state.pending_evaluations:
                raise EvaluationProtocolError(
                    f"request {request_id} is already pending"
                )
            seen_requests.add(request_id)
            candidate_ids = set(map(int, request.candidate_ids))
            if existing.intersection(candidate_ids):
                raise EvaluationProtocolError(
                    "a candidate is pending in another request"
                )
            estimated_cost = float(
                request.metadata.get("estimated_cost", len(candidate_ids))
            )
            if estimated_cost < 0 or not np.isfinite(estimated_cost):
                raise ValidationError("estimated_cost must be finite and non-negative")
            if (
                self.max_reserved_fe is not None
                and reserved_fe + len(candidate_ids) > self.max_reserved_fe
            ):
                raise EvaluationProtocolError(
                    "reserved evaluation budget would be exceeded"
                )
            if (
                self.max_reserved_cost is not None
                and fsum((reserved_cost, estimated_cost)) > self.max_reserved_cost
            ):
                raise EvaluationProtocolError("reserved cost budget would be exceeded")
            plans.append((request, estimated_cost))
            existing.update(candidate_ids)
            reserved_fe += len(candidate_ids)
            reserved_cost += estimated_cost
        handles = dict(state.evaluation_handles)
        owners = dict(state.evaluation_owners)
        pending_map = dict(state.pending_evaluations)
        started: list[tuple[int, Any]] = []
        try:
            for request, estimated_cost in plans:
                request_id = int(request.request_id)
                handle = self.evaluator.submit(request, state.problem)
                started.append((request_id, handle))
                pending = PendingEvaluation(
                    request,
                    EvaluationStatus.PENDING,
                    np.empty(0, dtype=np.int64),
                    reserved_cost=estimated_cost,
                    prediction=state.predictions,
                )
                checkpointable = self.evaluator.can_reattach(pending)
                pending = PendingEvaluation(
                    request,
                    EvaluationStatus.PENDING,
                    np.empty(0, dtype=np.int64),
                    reserved_cost=estimated_cost,
                    checkpointable=checkpointable,
                    prediction=state.predictions,
                )
                handles[request_id] = handle
                pending_map[request_id] = pending
                owners[request_id] = state.offspring
        except Exception as exc:
            cleanup_failed = []
            for _, handle in started:
                if not self.evaluator.cancel(handle) and not self.evaluator.detach(
                    handle
                ):
                    cleanup_failed.append(_)
            if cleanup_failed:
                partial = state.replace(
                    evaluation_handles=handles,
                    evaluation_owners=owners,
                    pending_evaluations=pending_map,
                )
                raise EvaluationSubmissionError(
                    "submission failed and started handles cannot be cleaned up: "
                    f"{cleanup_failed}",
                    partial,
                ) from exc
            raise
        if not requests:
            return state
        return state.replace(
            evaluation_handles=handles,
            evaluation_owners=owners,
            pending_evaluations=pending_map,
        )

    def cancel(self, state: OptimizationState, request_id: int) -> OptimizationState:
        """Cancel a pending request and commit its terminal update."""
        pending = state.pending_evaluations.get(request_id)
        handle = state.evaluation_handles.get(request_id)
        if pending is None or handle is None:
            raise EvaluationProtocolError(f"request {request_id} is not pending")
        if not self.evaluator.cancel(handle):
            raise EvaluationProtocolError(f"request {request_id} cannot be cancelled")
        update = EvaluationUpdate(
            np.int64(request_id),
            EvaluationStatus.CANCELLED,
            np.empty(0, dtype=np.int64),
            error=EvaluationErrorInfo("Cancelled", "evaluation cancelled"),
            sequence=pending.last_delivered_sequence + 1,
        )
        handle._delivered_sequence = update.sequence
        return self._commit_update(state, pending, handle, update)

    def poll(
        self, state: OptimizationState, *, wait: bool = False
    ) -> OptimizationState:
        """Collect completed work and commit updates serially."""
        if not state.pending_evaluations:
            return state
        for pending in state.pending_evaluations.values():
            if pending.fatal_error is not None:
                raise EvaluationFatalError(
                    pending.fatal_error.message,
                    state,
                )
        fatal = self._fatal_states.get(id(state))
        if fatal is not None:
            raise EvaluationFatalError(
                "async update has a persistent fatal state", fatal
            )
        current = state
        while current.pending_evaluations:
            progress = False
            ready: list[tuple[float, int, Any, EvaluationUpdate]] = []
            for request_id, handle in tuple(current.evaluation_handles.items()):
                pending = current.pending_evaluations.get(request_id)
                if pending is None:
                    continue
                timed_out = self.timeout is not None and (
                    time.monotonic() - handle._submitted_at >= self.timeout
                )
                if timed_out:
                    cancelled = self.evaluator.cancel(handle)
                    detached = False if cancelled else self.evaluator.detach(handle)
                    if not cancelled and not detached:
                        error = EvaluationErrorInfo(
                            "Timeout",
                            "evaluation timed out and backend cannot terminate",
                        )
                        pending_map = dict(current.pending_evaluations)
                        pending_map[request_id] = PendingEvaluation(
                            pending.request,
                            EvaluationStatus.CANCELLED,
                            pending.applied_candidate_ids,
                            pending.last_delivered_sequence,
                            pending.last_acknowledged_sequence,
                            {**pending.processing, -1: "timeout-fatal"},
                            pending.buffered_updates,
                            pending.reserved_cost,
                            pending.retry_count,
                            True,
                            pending.original_candidate_ids,
                            pending.feedback_result,
                            error,
                            pending.prediction,
                        )
                        data = dict(current.data)
                        data["async_fatal"] = {
                            "request_id": request_id,
                            "reason": error.message,
                        }
                        current = current.replace(
                            data=data,
                            pending_evaluations=pending_map,
                        )
                        progress = True
                        continue
                    update = EvaluationUpdate(
                        np.int64(request_id),
                        EvaluationStatus.CANCELLED,
                        np.empty(0, dtype=np.int64),
                        error=EvaluationErrorInfo("Timeout", "evaluation timed out"),
                        sequence=pending.last_delivered_sequence + 1,
                    )
                    handle._delivered_sequence = update.sequence
                    ready.append(
                        (
                            getattr(handle, "_completed_at", None) or time.monotonic(),
                            request_id,
                            handle,
                            update,
                        )
                    )
                    continue
                updates = self.evaluator.collect(handle, wait=False)
                for update in updates:
                    ready.append(
                        (
                            getattr(handle, "_completed_at", None) or time.monotonic(),
                            request_id,
                            handle,
                            update,
                        )
                    )
            for _, request_id, handle, update in sorted(
                ready, key=lambda item: (item[0], item[1], item[3].sequence)
            ):
                pending = current.pending_evaluations.get(request_id)
                if pending is None:
                    continue
                try:
                    current = self._commit_update(current, pending, handle, update)
                except EvaluationFatalError as exc:
                    self._fatal_states[id(state)] = exc.state
                    raise
                progress = True
            if not wait or not current.pending_evaluations:
                break
            if not progress:
                time.sleep(0.001)
        return current

    def checkpoint(self, state: OptimizationState, path: str) -> None:
        """Save pending state only when the evaluator can reattach it."""
        if any(
            pending.fatal_error is None and not self.evaluator.can_reattach(pending)
            for pending in state.pending_evaluations.values()
        ):
            raise CheckpointError("evaluator cannot reattach asynchronous pending work")
        state.save(path)

    def reattach(self, state: OptimizationState) -> OptimizationState:
        """Rebuild runtime handles for checkpointed requests."""
        handles = dict(state.evaluation_handles)
        for request_id, pending in state.pending_evaluations.items():
            if pending.fatal_error is not None:
                raise EvaluationFatalError(pending.fatal_error.message, state)
            if (
                pending.status
                in {
                    EvaluationStatus.COMPLETED,
                    EvaluationStatus.FAILED,
                    EvaluationStatus.CANCELLED,
                }
                and "callback-completed" in pending.processing.values()
            ):
                continue
            if not self.evaluator.can_reattach(pending):
                raise CheckpointError(f"evaluator cannot reattach request {request_id}")
            handles[request_id] = self.evaluator.reattach(pending, state.problem)
        current = state.replace(evaluation_handles=handles)
        for request_id, pending in tuple(current.pending_evaluations.items()):
            handle = current.evaluation_handles.get(request_id)
            for update in pending.buffered_updates:
                if pending.processing.get(update.sequence) == "committed":
                    continue
                current = self._commit_update(current, pending, handle, update)
                pending = current.pending_evaluations.get(request_id)
                if pending is None:
                    break
        return current

    def _commit_update(
        self,
        state: OptimizationState,
        pending: PendingEvaluation,
        handle: Any,
        update: EvaluationUpdate,
    ) -> OptimizationState:
        request_id = int(update.request_id)
        current_pending = state.pending_evaluations.get(request_id)
        if current_pending is None:
            return state
        original_ids = current_pending.original_candidate_ids
        if original_ids is None:
            raise EvaluationProtocolError("pending original candidate IDs are missing")
        processing = dict(current_pending.processing)
        key = int(update.sequence)
        if processing.get(key) == "committed":
            return state
        if processing.get(key) == "callback-completed":
            if update.status not in {
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }:
                return state
            pending_map = dict(state.pending_evaluations)
            pending_map.pop(request_id, None)
            handles = dict(state.evaluation_handles)
            handles.pop(request_id, None)
            owners = dict(state.evaluation_owners)
            owners.pop(request_id, None)
            return state.replace(
                pending_evaluations=pending_map,
                evaluation_handles=handles,
                evaluation_owners=owners,
            )
        if processing.get(key) in {"tell-started", "callback-started"}:
            if current_pending.fatal_error is not None:
                raise EvaluationFatalError(current_pending.fatal_error.message, state)
            raise EvaluationProtocolError(
                "update has a fatal post-effect failure and cannot be retried"
            )
        replay = key in processing
        if not replay and update.sequence != (
            current_pending.last_delivered_sequence + 1
        ):
            raise EvaluationProtocolError("async update sequence is not contiguous")
        if replay:
            previous = next(
                (
                    item
                    for item in current_pending.buffered_updates
                    if item.sequence == update.sequence
                ),
                None,
            )
            if previous is None or not self._same_update(previous, update):
                raise EvaluationProtocolError("redelivered update payload differs")
        else:
            self._validate_update(current_pending, update)
        buffered = list(current_pending.buffered_updates)
        if not any(
            item.request_id == update.request_id and item.sequence == update.sequence
            for item in buffered
        ):
            buffered.append(update)

        pending_record = current_pending

        def progress(
            current: OptimizationState,
            stage: str,
            applied: np.ndarray | None = None,
            fatal_error: EvaluationErrorInfo | None = None,
            acknowledged_sequence: int | None = None,
        ) -> OptimizationState:
            nonlocal current_pending, pending_record
            processing[key] = stage
            pending_record = PendingEvaluation(
                pending_record.request,
                update.status,
                pending_record.applied_candidate_ids if applied is None else applied,
                update.sequence,
                pending_record.last_acknowledged_sequence
                if acknowledged_sequence is None
                else acknowledged_sequence,
                processing.copy(),
                tuple(buffered),
                pending_record.reserved_cost,
                pending_record.retry_count,
                pending_record.checkpointable,
                original_ids,
                current.feedback_result,
                fatal_error,
                pending_record.prediction,
            )
            current_pending = pending_record
            return current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: pending_record,
                }
            )

        if not replay:
            current = state.replace(
                pending_evaluations={
                    **state.pending_evaluations,
                    request_id: PendingEvaluation(
                        current_pending.request,
                        update.status,
                        current_pending.applied_candidate_ids,
                        update.sequence,
                        current_pending.last_acknowledged_sequence,
                        {**processing, key: "received"},
                        tuple(buffered),
                        current_pending.reserved_cost,
                        current_pending.retry_count,
                        current_pending.checkpointable,
                        original_ids,
                        current_pending.feedback_result,
                        current_pending.fatal_error,
                        current_pending.prediction,
                    ),
                }
            )
        else:
            current = state
        current_pending = current.pending_evaluations[request_id]
        if current_pending.feedback_result is not None:
            current = current.replace(feedback_result=current_pending.feedback_result)
        has_rows = update.result is not None and len(update.candidate_ids) > 0
        ranks = {
            "received": 0,
            "population-applied": 1,
            "archived": 2,
            "feedback-applied": 3,
            "told": 4,
            "fe-applied": 5,
            "acknowledged": 6,
            "committed": 7,
            "callback-completed": 8,
        }
        stage_rank = ranks.get(processing.get(key, "received"), 0)
        if stage_rank < 1 and has_rows:
            current = self._apply_population(current, update)
            current = progress(current, "population-applied")
            stage_rank = 1
        if stage_rank < 2 and has_rows:
            current = self._apply_archive(current, update)
            current = progress(current, "archived")
            stage_rank = 2
        if stage_rank < 3 and has_rows:
            current = self._apply_feedback(current, update)
            current = progress(current, "feedback-applied")
            stage_rank = 3
        if stage_rank < 4 and has_rows:
            current = progress(
                current,
                "tell-started",
                fatal_error=EvaluationErrorInfo(
                    "TellFatal", "algorithm tell started and cannot be retried"
                ),
            )
            try:
                current = self._apply_tell(current, update)
            except Exception as exc:
                raise EvaluationFatalError(
                    "algorithm tell failed after side effects; update is fatal",
                    current,
                ) from exc
            current = progress(current, "told")
            stage_rank = 4
        if stage_rank < 5:
            if has_rows:
                current = current.replace(fe=current.fe + len(update.candidate_ids))
            current = progress(current, "fe-applied")
        if stage_rank < 6:
            self.evaluator.acknowledge(handle, update.sequence)
            current = progress(
                current,
                "acknowledged",
                acknowledged_sequence=update.sequence,
            )
        applied = np.unique(
            np.concatenate(
                [current_pending.applied_candidate_ids, update.candidate_ids]
            )
        )
        terminal = update.status in {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }
        remaining = np.setdiff1d(original_ids, applied)
        retried = False
        if (
            update.status is EvaluationStatus.FAILED
            and remaining.size
            and current_pending.retry_count < self.retry_limit
        ):
            retried = True
            rows = np.asarray(
                [
                    int(
                        np.flatnonzero(current_pending.request.candidate_ids == value)[
                            0
                        ]
                    )
                    for value in remaining
                ],
                dtype=np.intp,
            )
            request = EvaluationRequest(
                current_pending.request.request_id,
                remaining,
                current_pending.request.x[rows],
                current_pending.request.outputs,
                current_pending.request.metadata,
            )
            new_handle = self.evaluator.submit(request, state.problem)
            retry_pending = PendingEvaluation(
                request,
                EvaluationStatus.PENDING,
                applied,
                reserved_cost=current_pending.reserved_cost
                * len(remaining)
                / len(original_ids),
                retry_count=current_pending.retry_count + 1,
                checkpointable=self.evaluator.can_reattach(
                    PendingEvaluation(
                        request,
                        EvaluationStatus.PENDING,
                        applied,
                        checkpointable=False,
                    )
                ),
                original_candidate_ids=current_pending.original_candidate_ids,
                prediction=current_pending.prediction,
            )
            current = current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: retry_pending,
                },
                evaluation_handles={
                    **current.evaluation_handles,
                    request_id: new_handle,
                },
                evaluation_owners=current.evaluation_owners,
            )
        elif terminal:
            committed_pending = PendingEvaluation(
                current_pending.request,
                update.status,
                applied,
                update.sequence,
                update.sequence,
                {**processing, key: "committed"},
                tuple(buffered),
                current_pending.reserved_cost,
                current_pending.retry_count,
                current_pending.checkpointable,
                current_pending.original_candidate_ids,
                current_pending.feedback_result,
                None,
                current_pending.prediction,
            )
            current = current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: committed_pending,
                }
            )
        else:
            current_pending = PendingEvaluation(
                current_pending.request,
                update.status,
                applied,
                update.sequence,
                update.sequence,
                {**processing, key: "committed"},
                tuple(buffered),
                current_pending.reserved_cost,
                current_pending.retry_count,
                current_pending.checkpointable,
                original_ids,
                current_pending.feedback_result,
                None,
                current_pending.prediction,
            )
            current = current.replace(
                pending_evaluations={
                    **current.pending_evaluations,
                    request_id: current_pending,
                }
            )
        if len(update.candidate_ids) and self.callback_manager is not None:
            current = progress(
                current,
                "callback-started",
                applied=applied,
                fatal_error=EvaluationErrorInfo(
                    "CallbackFatal", "callback started and cannot be retried"
                ),
            )
            offspring = None
            owner = self._owner(current, request_id)
            offspring = owner.extract(
                self._rows(current, update.candidate_ids, request_id)
            )
            try:
                self.callback_manager.dispatch(
                    PostEvaluationEvent(
                        ctx=current,
                        offspring=offspring,
                        candidate_ids=update.candidate_ids,
                        request_id=update.request_id,
                        status=update.status,
                    )
                )
            except Exception as exc:
                raise EvaluationFatalError(
                    "post-evaluation callback failed; update is fatal", current
                ) from exc
            current = progress(current, "callback-completed", applied=applied)
        if terminal and not retried:
            pending_map = dict(current.pending_evaluations)
            pending_map.pop(request_id, None)
            handles = dict(current.evaluation_handles)
            handles.pop(request_id, None)
            owners = dict(current.evaluation_owners)
            owners.pop(request_id, None)
            current = current.replace(
                pending_evaluations=pending_map,
                evaluation_handles=handles,
                evaluation_owners=owners,
            )
        return current

    def _validate_update(
        self, pending: PendingEvaluation, update: EvaluationUpdate
    ) -> None:
        if int(update.request_id) != int(pending.request.request_id):
            raise EvaluationProtocolError(
                "update request_id does not match pending work"
            )
        requested = set(map(int, pending.request.candidate_ids))
        incoming = set(map(int, update.candidate_ids))
        applied = set(map(int, pending.applied_candidate_ids))
        if not incoming.issubset(requested) or incoming.intersection(applied):
            raise EvaluationProtocolError("async result candidate IDs are invalid")
        if update.result is not None and (
            update.result.candidate_ids is None
            or not np.array_equal(update.result.candidate_ids, update.candidate_ids)
        ):
            raise EvaluationProtocolError("result candidate IDs do not match update")
        if update.status in {EvaluationStatus.PENDING, EvaluationStatus.RUNNING} and (
            update.result is not None or len(update.candidate_ids)
        ):
            raise EvaluationProtocolError(
                "non-terminal running updates cannot carry results"
            )
        if update.status is EvaluationStatus.PARTIAL and (
            not len(update.candidate_ids) or update.result is None
        ):
            raise EvaluationProtocolError("partial updates require result rows")
        if update.status is EvaluationStatus.COMPLETED and (
            len(update.candidate_ids) > 0 and update.result is None
        ):
            raise EvaluationProtocolError("completed result rows require a result")
        if (
            update.status
            in {
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            }
            and update.error is None
        ):
            raise EvaluationProtocolError("terminal failure requires error details")
        original_ids = pending.original_candidate_ids
        if original_ids is None:
            raise EvaluationProtocolError("pending original candidate IDs are missing")
        original = set(map(int, original_ids))
        if update.status is EvaluationStatus.COMPLETED and (
            incoming | applied != original
        ):
            raise EvaluationProtocolError(
                "completed update does not account for all requested candidates"
            )
        if pending.status in {
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
            EvaluationStatus.CANCELLED,
        }:
            raise EvaluationProtocolError("terminal evaluation cannot transition")
        allowed = {
            EvaluationStatus.PENDING: {
                EvaluationStatus.RUNNING,
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            EvaluationStatus.RUNNING: {
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
            EvaluationStatus.PARTIAL: {
                EvaluationStatus.PARTIAL,
                EvaluationStatus.COMPLETED,
                EvaluationStatus.FAILED,
                EvaluationStatus.CANCELLED,
            },
        }
        if update.status not in allowed.get(pending.status, set()):
            raise EvaluationProtocolError("invalid evaluation status transition")

    @staticmethod
    def _same_update(left: EvaluationUpdate, right: EvaluationUpdate) -> bool:
        if (
            left.request_id != right.request_id
            or left.status is not right.status
            or left.sequence != right.sequence
            or not np.array_equal(left.candidate_ids, right.candidate_ids)
        ):
            return False
        if (left.error is None) != (right.error is None):
            return False
        if (
            left.error is not None
            and right.error is not None
            and (
                left.error.error_type != right.error.error_type
                or left.error.message != right.error.message
                or dict(left.error.details) != dict(right.error.details)
            )
        ):
            return False
        if (left.result is None) != (right.result is None):
            return False
        if left.result is None or right.result is None:
            return True
        return (
            np.array_equal(left.result.f, right.result.f)
            and np.array_equal(left.result.g, right.result.g)
            and np.array_equal(left.result.cv, right.result.cv)
            and AsyncScheduler._optional_array_equal(
                left.result.candidate_ids, right.result.candidate_ids
            )
            and AsyncScheduler._optional_array_equal(
                left.result.cost, right.result.cost
            )
            and AsyncScheduler._optional_array_equal(
                left.result.noise, right.result.noise
            )
            and left.result.outputs.keys() == right.result.outputs.keys()
            and all(
                np.array_equal(left.result.outputs[name], right.result.outputs[name])
                for name in left.result.outputs
            )
        )

    @staticmethod
    def _optional_array_equal(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return bool(np.array_equal(left, right))

    def _owner(self, state: OptimizationState, request_id: int) -> Any:
        owner = state.evaluation_owners.get(request_id, state.offspring)
        if owner is None:
            raise EvaluationProtocolError("async evaluation owner is missing")
        return owner

    def _rows(
        self, state: OptimizationState, ids: np.ndarray, request_id: int
    ) -> np.ndarray:
        owner = self._owner(state, request_id)
        if "id" not in owner.schema:
            raise EvaluationProtocolError("async updates require offspring IDs")
        source = owner.get_array("id")
        rows = [np.flatnonzero(source == value) for value in ids]
        if any(len(row) != 1 for row in rows):
            raise EvaluationProtocolError("async result candidate is not in offspring")
        return np.asarray([int(row[0]) for row in rows], dtype=np.intp)

    def _apply_population(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if update.result is None:
            return state
        owner = self._owner(state, int(update.request_id))
        rows = self._rows(state, update.candidate_ids, int(update.request_id))
        owner.update_rows(
            rows,
            {
                name: value
                for name, value in {
                    "f": update.result.f,
                    "g": update.result.g,
                    "cv": update.result.cv,
                }.items()
                if name in owner.schema
            },
        )
        return state

    def _apply_archive(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if update.result is None:
            return state
        owner = self._owner(state, int(update.request_id))
        rows = self._rows(state, update.candidate_ids, int(update.request_id))
        archives = (state.archive, state.pareto_archive)

        def values_for(archive: Any) -> list[dict[str, Any]]:
            values_list = []
            for row, candidate_id in zip(rows, update.candidate_ids, strict=True):
                values = {
                    name: owner.get_array(name)[row]
                    for name in archive.schema
                    if name in owner.schema
                }
                if "id" in archive.schema:
                    values["id"] = np.int64(candidate_id)
                if "request_id" in archive.schema:
                    values["request_id"] = np.int64(update.request_id)
                values_list.append(values)
            return values_list

        snapshots = []
        for archive in archives:
            snapshots.append(
                (
                    archive,
                    {
                        name: np.array(array, copy=True)
                        for name, array in archive._data.items()
                    },
                    archive._size,
                    archive._structure_version,
                    archive._value_version,
                    dict(archive._cache),
                    getattr(archive, "_kdtree", None),
                    list(getattr(archive, "_deprecated_duplicate_indices", [])),
                )
            )
        try:
            for archive in archives:
                for values in values_for(archive):
                    if "id" in values and len(archive):
                        ids = archive.get_array("id")[: len(archive)]
                        if np.any(ids == values["id"]):
                            policy = getattr(archive, "duplicate_policy", "keep_first")
                            if (
                                policy == "keep_first"
                                and archive is not state.pareto_archive
                            ):
                                continue
                            if policy == "append":
                                request_ids = archive.get_array("request_id")[
                                    : len(archive)
                                ]
                                if np.any(
                                    (ids == values["id"])
                                    & (request_ids == values.get("request_id"))
                                ):
                                    continue
                            else:
                                archive.delete(np.flatnonzero(ids == values["id"]))
                    archive.add(values)
        except Exception:
            for (
                archive,
                snapshot,
                size,
                structure_version,
                value_version,
                cache,
                kdtree,
                duplicate_indices,
            ) in snapshots:
                archive._data = snapshot
                archive._size = size
                archive._structure_version = structure_version
                archive._value_version = value_version
                archive._cache = cache
                if hasattr(archive, "_kdtree"):
                    archive._kdtree = kdtree
                if hasattr(archive, "_deprecated_duplicate_indices"):
                    archive._deprecated_duplicate_indices = duplicate_indices
            raise
        return state

    def _apply_feedback(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if self.feedback_policy is None or update.result is None:
            return state
        evaluation = update.result
        owner = self._owner(state, int(update.request_id))
        result: FeedbackResult = self.feedback_policy.build(
            owner,
            state.pending_evaluations[int(update.request_id)].prediction
            or state.predictions,
            evaluation,
            update.candidate_ids,
            state,
        )
        if len(result.candidate_ids):
            rows = self._rows(state, result.candidate_ids, int(update.request_id))
            values = {"f": result.f}
            if result.g is not None and "g" in owner.schema:
                values["g"] = result.g
            if result.cv is not None and "cv" in owner.schema:
                values["cv"] = result.cv
            owner.update_rows(rows, values)
        return state.replace(feedback_result=result)

    def _apply_tell(
        self, state: OptimizationState, update: EvaluationUpdate
    ) -> OptimizationState:
        if (
            self.algorithm is None
            or state.feedback_result is None
            or len(state.feedback_result.candidate_ids) == 0
        ):
            return state
        owner = self._owner(state, int(update.request_id))
        rows = self._rows(
            state, state.feedback_result.candidate_ids, int(update.request_id)
        )
        self.algorithm.tell(state, self, owner.extract(rows))
        return state
