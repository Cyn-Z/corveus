"""Explicit, single-inference worker coordinator; importing starts no work.

The caller supplies trusted preparation, memory resolution, a byte reservation,
and a synchronous non-streaming model callback. A normal callback return must
prove that upstream execution completed and cannot execute again. Timeouts and
exceptions never constitute stop confirmation. No HTTP or deployment wiring is
installed here.
"""

from __future__ import annotations

from threading import Event, Lock, Thread
from typing import Callable

from analyst_generation import (
    InferenceCallUncertainError,
    InvalidInferenceResponseError,
    generate_answer,
)
from service_contracts import ErrorCode, JobState, JobStatus, ServiceError
from service_preparation import JobPreparation
from service_results import JobResults, ResultSnapshot
from service_scheduler import WorkerLease, _identifier


class _LeaseRenewal:
    """Renew during blocking preparation/tokenization/model calls.

    An error permanently revokes this helper's dispatch permission. This does
    not stop an already dispatched upstream request or release its capacity.
    """

    def __init__(self, scheduler, claim):
        self.scheduler = scheduler
        self.claim = claim
        self.done = Event()
        self.failed = Event()
        self.thread = Thread(target=self._run, name="service-lease", daemon=True)

    def _run(self):
        while not self.done.wait(self.scheduler.store.settings.heartbeat_seconds):
            try:
                self.scheduler.heartbeat(self.claim)
            except Exception:
                self.failed.set()
                return

    def check(self) -> WorkerLease:
        if self.failed.is_set():
            raise ServiceError(ErrorCode.CONFLICT)
        return self.scheduler.heartbeat(self.claim)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()


class JobWorker:
    """Run one claimed job at a time, with no automatic inference retries.

    All collaborators must share the same scheduler instance. ``reserved_bytes``
    is a trusted server estimate, not available VRAM or a client input. This
    coordinator requires the shared inference limit to be exactly one.

    ``run_once`` recovers durable finalizations before claiming fresh work.
    Storage errors propagate so a supervisor can back off. Restarting cannot
    redispatch running/interrupted jobs. Uncertain holds require separate trusted
    upstream verification and InferenceGate.reconcile_stopped.
    """

    def __init__(
        self,
        worker_id: str,
        preparation: JobPreparation,
        results: JobResults,
        *,
        call_llm: Callable,
        reserved_bytes: int,
        poll_seconds: float = 1.0,
    ) -> None:
        _identifier(worker_id, "worker_id")
        if not isinstance(preparation, JobPreparation):
            raise ValueError("preparation must be JobPreparation.")
        if not isinstance(results, JobResults):
            raise ValueError("results must be JobResults.")
        if preparation.scheduler is not results.scheduler:
            raise ValueError("Collaborators must share the same scheduler.")
        if preparation.store.settings.max_inference_requests != 1:
            raise ValueError("This worker requires inference concurrency one.")
        if not callable(call_llm):
            raise ValueError("call_llm must be callable.")
        if type(reserved_bytes) is not int or not 1 <= reserved_bytes < 2**63:
            raise ValueError("reserved_bytes must be a positive SQLite integer.")
        budget = preparation.store.settings.inference_memory_budget_bytes
        if budget is not None and reserved_bytes > budget:
            raise ValueError("Reservation exceeds the configured memory budget.")
        if type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= 60:
            raise ValueError("poll_seconds must be between zero and 60 seconds.")
        self.worker_id = worker_id
        self.preparation = preparation
        self.results = results
        self.scheduler = preparation.scheduler
        self.store = preparation.store
        self.gate = results.gate
        self.call_llm = call_llm
        self.reserved_bytes = reserved_bytes
        self.poll_seconds = poll_seconds
        self._active = Lock()

    def _finish_stopped(self, permit):
        """Release a proven stopped call, even if the lease was lost."""
        try:
            return self.gate.finish(permit, inference_stopped=True, failed=True)
        except ServiceError as exc:
            if exc.code is not ErrorCode.CONFLICT:
                raise
            self.gate.interrupt(permit)
            return self.gate.reconcile_stopped(permit, inference_stopped=True)

    def run_once(self, stop: Event | None = None) -> JobStatus | None:
        """Recover results and process at most one job; None means no claim.

        Shutdown before dispatch returns preparation to the queue. After
        dispatch, shutdown drains the synchronous call while renewing its lease;
        it never pretends to cancel upstream execution. Use a unique worker ID
        for each process. Concurrent calls on this instance are rejected.
        """
        if stop is None:
            stop = Event()
        if not isinstance(stop, Event):
            raise ValueError("stop must be a threading.Event.")
        if not self._active.acquire(blocking=False):
            raise RuntimeError("This worker is already processing a job.")
        try:
            if stop.is_set():
                return None
            self.scheduler.recover()
            self.results.recover()
            claim = self.scheduler.claim(
                self.worker_id,
                profile_ids=(self.preparation.preparer.profile.profile_id,),
            )
            if claim is None:
                return None
            return self._process(claim, stop)
        finally:
            self._active.release()

    def _process(self, claim, stop):
        permit = None
        dispatched = False
        stopped = False
        captured = False
        try:
            with _LeaseRenewal(self.scheduler, claim) as renewal:
                status = self.preparation.prepare(claim)
                if status.state is not JobState.WAITING_INFERENCE:
                    return status
                snapshot = self.preparation.for_worker(claim)
                if snapshot is None or snapshot.needs_review:
                    raise ServiceError(ErrorCode.INTERNAL_ERROR)
                while permit is None:
                    renewal.check()
                    if stop.is_set():
                        return self.scheduler.release(claim)
                    permit = self.gate.start(claim, reserved_bytes=self.reserved_bytes)
                    if permit is None:
                        stop.wait(self.poll_seconds)

                def dispatch(messages):
                    nonlocal dispatched, stopped
                    live = renewal.check()
                    if stop.is_set() or live.status.cancel_requested:
                        # generate_answer classifies all callback errors as
                        # uncertain; dispatched remains False for this case.
                        raise ServiceError(ErrorCode.CONFLICT)
                    dispatched = True
                    response = self.call_llm(messages)
                    stopped = True
                    return response

                profile = snapshot.profile
                output = generate_answer(
                    snapshot.messages(),
                    count_input_tokens=self.preparation.preparer.count_input_tokens,
                    call_llm=dispatch,
                    model_name=profile.model_id,
                    context_size=profile.context_size,
                    input_budget=profile.input_token_budget,
                )
                status = self.results.capture(
                    permit, ResultSnapshot(output.content, output.usage),
                    inference_stopped=True,
                )
                captured = True
            if status.state is JobState.FINALIZING:
                return self.results.finalize(
                    status.principal, status.job_id,
                ).status
            return status
        except (InferenceCallUncertainError, InvalidInferenceResponseError):
            if not dispatched or stopped:
                return self._finish_stopped(permit)
            return self.gate.interrupt(permit)
        except ServiceError as exc:
            if captured:
                # The durable result must remain available for recovery.
                raise
            if permit is not None:
                if not dispatched or stopped:
                    status = self._finish_stopped(permit)
                else:
                    status = self.gate.interrupt(permit)
            elif exc.code is ErrorCode.QUEUE_FULL:
                return self.scheduler.release(claim)
            else:
                try:
                    status = self.scheduler.fail(claim)
                except ServiceError as cleanup_error:
                    if cleanup_error.code is not ErrorCode.CONFLICT:
                        raise
                    self.scheduler.recover()
                    status = self.store.get(
                        claim.status.principal, claim.status.job_id,
                    )
            if exc.code in (
                ErrorCode.CONFLICT, ErrorCode.CONTENT_EXPIRED,
                ErrorCode.DEADLINE_EXCEEDED,
            ):
                return status
            raise
        except BaseException:
            # Never release uncertainty, including process interruption during
            # dispatch. A captured answer belongs to finalization recovery.
            if permit is not None and not captured:
                if not dispatched or stopped:
                    self._finish_stopped(permit)
                else:
                    self.gate.interrupt(permit)
            elif permit is None:
                try:
                    self.scheduler.fail(claim)
                except ServiceError as exc:
                    if exc.code is not ErrorCode.CONFLICT:
                        raise
            raise

    def run_forever(self, stop: Event) -> None:
        """Poll explicitly until shutdown; errors propagate to the supervisor."""
        if not isinstance(stop, Event):
            raise ValueError("stop must be a threading.Event.")
        while not stop.is_set():
            self.run_once(stop)
            stop.wait(self.poll_seconds)
