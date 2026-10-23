"""Explicit startup for separate local API and worker processes."""

from __future__ import annotations

import argparse
import os
import signal
import time
import uuid
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from analyst_preparation import PromptPreparer
from service_contracts import ExecutionProfile, HardwareTarget
from service_runtime import build_local_service
from service_settings import load_service_settings
from service_health import WorkerHealth


def unavailable(*args, **kwargs):
    raise RuntimeError("This operation is unavailable in the API process.")


def build_runtime(role):
    if role not in {"api", "worker"}:
        raise ValueError("role must be api or worker.")

    service_settings = load_service_settings()
    if service_settings.hardware_target is not HardwareTarget.GENERIC:
        raise ValueError("This deployment batch supports the RTX 3060 only.")
    if service_settings.max_inference_requests != 1:
        raise ValueError("Inference concurrency must remain one.")

    budget = service_settings.inference_memory_budget_bytes
    if budget is None:
        raise ValueError(
            "SERVICE_INFERENCE_MEMORY_BUDGET_BYTES must be configured."
        )

    database_path = Path(
        os.environ.get(
            "SERVICE_JOB_DATABASE_PATH", "/app/data/jobs.sqlite3"
        )
    )
    if not database_path.is_absolute():
        raise ValueError("SERVICE_JOB_DATABASE_PATH must be absolute.")
    if not database_path.parent.is_dir():
        raise ValueError("The job database directory must already exist.")

    # Set before importing analyst, which initializes application components.
    os.environ["SERVICE_PROCESS_ROLE"] = role
    import analyst

    if database_path.resolve() == (
        analyst.settings.memory.database_path.resolve()
    ):
        raise ValueError("Job and conversation databases must be separate.")

    profile = ExecutionProfile(
        profile_id="default",
        model_id=analyst.settings.model_name,
        inference_pool_id="local-gpu",
        context_size=analyst.settings.context_size,
        max_output_tokens=analyst.settings.max_output_tokens,
        safety_margin=analyst.settings.context_safety_margin,
        hardware_target=service_settings.hardware_target,
    )

    if role == "worker":
        router = analyst.SKILL_ROUTER
        registry = analyst.SKILL_REGISTRY
        call_llm = analyst.call_llm
    else:
        # API review can recount saved instructions, but cannot reroute
        # prompts or dispatch generation.
        router = SimpleNamespace(search=unavailable)
        registry = SimpleNamespace(load_instructions=unavailable)
        call_llm = unavailable

    preparer = PromptPreparer(
        router,
        registry,
        analyst.count_input_tokens,
        profile,
        top_k=analyst.settings.router_top_k,
    )
    runtime = build_local_service(
        database_path,
        service_settings,
        preparer,
        worker_id=f"{role}-{uuid.uuid4().hex}",
        call_llm=call_llm,
        # Reserve the whole configured accounting pool for each request.
        # This is not a measurement or guarantee of physical VRAM use.
        reserved_bytes=budget,
        memory=analyst.MEMORY_MANAGER,
    )
    return runtime, analyst.MEMORY_MANAGER


def serve_api(runtime, stop):
    from http.server import HTTPServer
    from http_server import OrchestratorHandler, attach_local_job_api

    server = HTTPServer(("0.0.0.0", 5000), OrchestratorHandler)
    attach_local_job_api(server, runtime.api)
    server.timeout = 1
    try:
        while not stop.is_set():
            server.handle_request()
    finally:
        server.server_close()


def serve_worker(runtime, memory, stop):
    next_cleanup = 0.0
    while not stop.is_set():
        runtime.worker.run_once(stop)

        now = time.monotonic()
        if now >= next_cleanup:
            runtime.worker.preparation.requests.prune_expired()
            memory.prune_expired_conversations()
            next_cleanup = now + 60

        stop.wait(1)

def wait_for_model(stop):
    import urllib.request
    from urllib.parse import urlsplit
    import analyst

    url = urlsplit(analyst.settings.llm_url)
    health_url = f"{url.scheme}://{url.netloc}/health"

    print("Waiting for the inference server to become ready.", flush=True)
    while not stop.is_set():
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return True
        except Exception:
            # Readiness failure grants no inference permission and
            # confirms nothing about previously interrupted execution.
            pass
        stop.wait(1)
    return False

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("api", "worker"))
    arguments = parser.parse_args(argv)

    stop = Event()

    def request_shutdown(signum, frame):
        # A running worker drains its synchronous call. This does not
        # confirm termination of uncertain upstream inference.
        stop.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    runtime, memory = build_runtime(arguments.role)
    if arguments.role == "api":
        serve_api(runtime, stop)
    else:
        with WorkerHealth(runtime.worker.store.path).heartbeat(
            runtime.worker.worker_id
        ):
            if wait_for_model(stop):
                serve_worker(runtime, memory, stop)


if __name__ == "__main__":
    main()
