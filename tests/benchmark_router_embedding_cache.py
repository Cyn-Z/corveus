"""Opt-in real-model comparison. Never invoked by unittest discovery.

Run only after approval. Uses temporary caches, offline models, and synthetic
cases. Does not start workers, call inference, or access conversation storage.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def subprocess_environment():
    # BM25 currently sums terms from a set. Fix process hash order for exact
    # numeric comparisons; this is a benchmark control, not a runtime change.
    seed = os.environ.get("PYTHONHASHSEED", "0")
    if seed == "random":
        seed = "0"
    return {**os.environ, "PYTHONHASHSEED": seed, "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def input_fingerprints(repo, model):
    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                result.update(chunk)
        return result.hexdigest()

    repo, model = repo.resolve(), model.resolve()
    index = json.loads((repo / "index.json").read_text())
    instructions = {}
    for entry in index.get("skills", []):
        path = (repo / entry.get("path", "") / "SKILL.md").resolve()
        if path.is_relative_to(repo) and path.is_file():
            instructions[path.relative_to(repo).as_posix()] = digest(path)
    return {"catalog_index_sha256": digest(repo / "index.json"),
            "instructions": instructions,
            "model_artifacts": {path.relative_to(model).as_posix(): digest(path)
                                for path in sorted(model.rglob("*")) if path.is_file()}}


def fingerprint_source(source):
    digest = hashlib.sha256()
    files = {}
    for name in ("orchestrator.py", "router_policy.py", "router_query_plan.py",
                 "router_long_requests.py", "router_evaluation.py", "router_embedding_cache.py"):
        path = source / "analyst" / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest.update(json.dumps(files, sort_keys=True).encode())
    return {"sha256": digest.hexdigest(), "files": files}


def probe(args):
    settings = json.loads(Path(args.settings_json).read_text())
    import_start = time.perf_counter()
    sys.path.insert(0, str(Path(args.source) / "analyst"))
    import orchestrator
    import router_evaluation as evaluation
    from router_policy import RouterPolicy, RouterSelectionPolicy
    from router_query_plan import QueryPlanSettings
    import_seconds = time.perf_counter() - import_start
    calls = {"catalog": 0, "query": 0}
    loader = orchestrator.SkillRouter._load_embedding_model

    def counted_loader(path):
        model = loader(path)
        encode = model.encode

        def counted_encode(text, *positional, **keywords):
            calls["catalog" if isinstance(text, list) else "query"] += 1
            return encode(text, *positional, **keywords)

        model.encode = counted_encode
        return model

    orchestrator.SkillRouter._load_embedding_model = staticmethod(counted_loader)
    kwargs = {
        "model_path": args.model_path,
        "semantic_weight": settings["semantic_weight"],
        "name_boost": settings["name_boost"],
        "activation_policy": RouterPolicy(**settings["activation_policy"]),
        "selection_policy": RouterSelectionPolicy(**settings["selection_policy"]),
        "query_plan_settings": QueryPlanSettings(**settings["query_plan"]),
    }
    supports_cache = "embedding_cache_settings" in inspect.signature(orchestrator.SkillRouter).parameters
    if supports_cache:
        from router_embedding_cache import EmbeddingCacheSettings
        kwargs["embedding_cache_settings"] = EmbeddingCacheSettings(
            enabled=args.mode != "disabled", directory=Path(args.cache_dir)
        )
    elif args.mode != "disabled":
        raise ValueError("This source does not implement embedding caching")
    started = time.perf_counter()
    with redirect_stdout(sys.stderr):
        registry = orchestrator.SkillRegistry(args.repo)
        router = orchestrator.SkillRouter(registry, **kwargs)
    initialization_seconds = time.perf_counter() - started
    result = {
        "mode": args.mode,
        "initialization_seconds": initialization_seconds,
        "source_import_seconds": import_seconds,
        "import_and_initialization_seconds": import_seconds + initialization_seconds,
        "startup_embedding_calls": dict(calls),
        "cache": getattr(router, "embedding_cache_report", {"status": "unimplemented"}),
    }
    if args.full_evaluation:
        import numpy as np
        from importlib.metadata import PackageNotFoundError, version

        cases = {case.id: case for split in evaluation.SPLITS.values() for case in split}
        task = "Investigate Windows Event 4625 failed logons using Splunk.\n"
        log = "heartbeat status=ok\n" * 300
        cases["cache-long-middle"] = evaluation.RoutingCase(
            "cache-long-middle", "long-input", log + task + log, True, (evaluation.WINDOWS_LOGS,)
        )
        cases = tuple(cases.values())
        top = settings["top_k"]
        before = dict(calls)
        result["evaluation"] = evaluation.evaluate_router(
            router, cases, limit=top, measure_instructions=True
        )
        result["evaluation_embedding_calls"] = {
            key: calls[key] - before[key] for key in before
        }
        result["vector_sha256"] = hashlib.sha256(
            np.asarray(router.embedding_matrix, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        result["ordered_skill_names"] = [skill.name for skill in registry.skills]
        result["selection_scores"] = {}
        result["all_candidate_scores_sha256"] = {}
        for case in cases:
            result["selection_scores"][case.id] = [
                [skill.name, score] for score, skill in router.search(case.prompt, limit=top)
            ]
            # Explicitly exercise legacy scalar-boundary guards and full ranking.
            records = [asdict(score) for score in router.score_candidates(case.prompt, len(registry.skills))]
            result["all_candidate_scores_sha256"][case.id] = hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        result["benchmarks"] = {}
        for interface in ("search", "route"):
            call = getattr(router, interface)
            measurements = []
            for case in cases:
                for _ in range(args.warmup):
                    call(case.prompt, limit=top)
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    call(case.prompt, limit=top)
                    samples.append(time.perf_counter() - start)
                measurements.append({"case_id": case.id, **evaluation.latency_summary(samples)})
            result["benchmarks"][interface] = measurements
        result["source"] = fingerprint_source(Path(args.source))
        result["cases_sha256"] = result["evaluation"]["cases_sha256"]
        result["model_sha256"] = evaluation.fingerprint_model(args.model_path)
        result["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
        result["environment"]["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
        result["environment"]["thread_environment"] = {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        }
        result["environment"]["packages"] = {}
        for package in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers"):
            try:
                result["environment"]["packages"][package] = version(package)
            except PackageNotFoundError:
                result["environment"]["packages"][package] = None
        result["effective_settings"] = settings
    print(json.dumps(result, allow_nan=False))


def summary(values):
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {"samples": len(values), "median_ms": median * 1000,
            "p95_ms": ordered[__import__("math").ceil(.95 * len(ordered)) - 1] * 1000,
            "median_absolute_deviation_ms": statistics.median(abs(value - median) for value in values) * 1000}


def compare(args):
    sources = {"baseline": Path(args.baseline_source).resolve(),
               "disabled": Path(args.candidate_source).resolve(),
               "build": Path(args.candidate_source).resolve(),
               "hit": Path(args.candidate_source).resolve()}
    reference_path = sources["baseline"] / "reference-inputs.json"
    if not reference_path.is_file():
        raise ValueError("Baseline source requires reference-inputs.json")
    input_reference = json.loads(reference_path.read_text())
    current_inputs = input_fingerprints(Path(args.repo), Path(args.model_path))
    if current_inputs != input_reference:
        raise ValueError("Catalog, instructions, or model artifacts differ from the preserved baseline inputs")
    samples = {name: [] for name in sources}
    details = {}
    with tempfile.TemporaryDirectory(prefix="corveus-cache-measurement-") as temporary:
        root = Path(temporary)
        for index in range(args.runs):
            cache_dir = root / f"cache-{index}"
            # Alternate disabled sources; build must always precede its cache hit.
            order = ("baseline", "disabled", "build", "hit") if index % 2 == 0 else (
                "build", "hit", "disabled", "baseline")
            for name in order:
                command = [
                    sys.executable, "-B", str(Path(__file__).resolve()), "--probe",
                    "--source", str(sources[name]), "--repo", str(Path(args.repo).resolve()),
                    "--model-path", str(Path(args.model_path).resolve()),
                    "--settings-json", str(Path(args.settings_json).resolve()),
                    "--mode", "disabled" if name == "baseline" else name,
                    "--cache-dir", str(cache_dir), "--repeats", str(args.repeats),
                    "--warmup", str(args.warmup),
                ]
                if index == 0:
                    command.append("--full-evaluation")
                result = subprocess.run(command, check=True, capture_output=True, text=True,
                                        timeout=900, env=subprocess_environment())
                data = json.loads(result.stdout)
                # Model evaluation after startup is excluded from all startup samples.
                samples[name].append({key: data[key] for key in (
                    "initialization_seconds", "source_import_seconds", "import_and_initialization_seconds",
                    "startup_embedding_calls", "cache")})
                if index == 0:
                    details[name] = data
                print(f"Startup sample {index + 1}/{args.runs}: {name}", file=sys.stderr, flush=True)
    reference = details["baseline"]
    checks = {}
    fields = ("vector_sha256", "ordered_skill_names", "selection_scores",
              "all_candidate_scores_sha256", "evaluation_embedding_calls", "cases_sha256",
              "effective_settings", "model_sha256", "environment")
    evaluation_fields = ("actual_activation", "routing_reason", "selected_skills",
                         "required_groups_hit", "required_groups_total", "selection_judgments",
                         "redundant_selected", "selected_instruction_characters",
                         "selected_instruction_embedding_tokens")
    for name in ("disabled", "build", "hit"):
        checks[name] = {field: reference[field] == details[name][field] for field in fields}
        checks[name]["evaluation"] = (
            [{key: row[key] for key in evaluation_fields} for row in reference["evaluation"]["cases"]]
            == [{key: row[key] for key in evaluation_fields} for row in details[name]["evaluation"]["cases"]]
        )
    calls_ok = all(
        sample["startup_embedding_calls"] == {"catalog": 0 if name == "hit" else 1, "query": 0}
        for name, rows in samples.items() for sample in rows
    )
    cache_states_ok = all(
        sample["cache"]["status"] == ("built" if name == "build" else "hit")
        for name in ("build", "hit") for sample in samples[name]
    )
    startup = {
        name: {field: summary([row[field] for row in rows]) for field in (
            "initialization_seconds", "source_import_seconds", "import_and_initialization_seconds")}
        for name, rows in samples.items()
    }
    disabled = startup["disabled"]["import_and_initialization_seconds"]
    hit = startup["hit"]["import_and_initialization_seconds"]
    improvement = disabled["median_ms"] - hit["median_ms"]
    noise = max(disabled["median_absolute_deviation_ms"], hit["median_absolute_deviation_ms"])
    if input_fingerprints(Path(args.repo), Path(args.model_path)) != input_reference:
        raise ValueError("Baseline inputs changed during measurement; discard this run")
    report = {"input_reference": input_reference,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parity": checks, "embedding_calls_correct": calls_ok, "cache_states_correct": cache_states_ok,
              "startup": startup, "startup_samples": samples, "details": details,
              "performance": {"median_improvement_ms": improvement, "observed_noise_ms": noise,
                              "improvement_exceeds_observed_noise": improvement > noise},
              "scope": "Real CPU embedding model; startup includes imports, excludes interpreter launch; OS page caches uncontrolled. Warm routing excludes instructions and inference."}
    report["passed"] = all(all(values.values()) for values in checks.values()) and calls_ok and cache_states_ok
    output = Path(args.output).resolve()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Report: {output}; correctness passed={report['passed']}", file=sys.stderr)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source")
    parser.add_argument("--candidate-source")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("disabled", "build", "hit"), default="disabled", help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", help=argparse.SUPPRESS)
    parser.add_argument("--full-evaluation", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("runs/repeats must be positive and warmup nonnegative")
    if args.probe:
        if not args.source or not args.cache_dir:
            parser.error("probe requires source and cache directory")
        probe(args)
        return 0
    if not args.baseline_source or not args.candidate_source or not args.output:
        parser.error("baseline-source, candidate-source, and output are required")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new report filename")
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
