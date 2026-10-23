"""Explicit composition for the future local Web/API and worker bootstrap.

No model imports, threads, environment changes, or automatic startup. Separate
processes may construct equivalent runtimes over the same local SQLite file.
"""

from dataclasses import dataclass

from analyst_preparation import PromptPreparer
from memory_manager import MemoryManager
from service_api import LocalJobAPI
from service_inference import InferenceGate
from service_job_store import JobStore
from service_preparation import JobPreparation
from service_results import JobResults
from service_scheduler import JobScheduler
from service_settings import ServiceSettings
from service_worker import JobWorker


@dataclass(frozen=True)
class LocalServiceRuntime:
    api: LocalJobAPI
    worker: JobWorker


def build_local_service(
    database_path,
    settings: ServiceSettings,
    preparer: PromptPreparer,
    *,
    worker_id: str,
    call_llm,
    reserved_bytes: int,
    memory: MemoryManager | None = None,
) -> LocalServiceRuntime:
    """Compose one fixed operator scope from trusted server dependencies.

    ``memory`` must already represent that operator's configured conversation
    scope. No mapping from arbitrary principals to this manager is provided.
    The callback must follow JobWorker's no-retry, non-streaming contract.
    """
    if not isinstance(settings, ServiceSettings):
        raise ValueError("settings must be ServiceSettings.")
    if not isinstance(preparer, PromptPreparer):
        raise ValueError("preparer must be PromptPreparer.")
    if memory is not None and not isinstance(memory, MemoryManager):
        raise ValueError("memory must be MemoryManager or None.")
    if settings.max_inference_requests != 1:
        raise ValueError("Local service requires inference concurrency one.")
    if preparer.profile.hardware_target != settings.hardware_target:
        raise ValueError("Preparation and service hardware targets must match.")
    store = JobStore(database_path, settings)
    scheduler = JobScheduler(store)
    gate = InferenceGate(scheduler)

    def memory_for(principal):
        return memory if principal == settings.local_principal else None

    results = JobResults(gate, memory_for=memory_for)
    preparation = JobPreparation(scheduler, preparer)
    worker = JobWorker(
        worker_id, preparation, results,
        call_llm=call_llm, reserved_bytes=reserved_bytes,
    )
    api = LocalJobAPI(
        results,
        profile_id=preparer.profile.profile_id,
        memory=memory,
        preparation=preparation,
    )
    return LocalServiceRuntime(api=api, worker=worker)