"""Offline, synthetic routing benchmark; desired labels are reviewable policy.

This measures activation and expected skill-group coverage separately. It
does not claim that other returned skills are necessarily irrelevant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import platform
import subprocess
import os
from importlib.metadata import version
from pathlib import Path
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass

from orchestrator import SkillRegistry, SkillRouter
from router_policy import RouterPolicy, RouterSelectionPolicy
from router_query_plan import (
    add_query_plan_arguments,
    query_plan_settings_from_arguments,
)


@dataclass(frozen=True)
class RoutingCase:
    id: str
    category: str
    prompt: str
    expect_skills: bool
    # Each group is a set of acceptable alternatives. Multi-topic cases
    # require at least one selected skill from every group.
    required_groups: tuple[tuple[str, ...], ...] = ()
    irrelevant_skills: tuple[str, ...] = ()
    redundancy_groups: tuple[tuple[str, ...], ...] = ()


PHISHING = (
    "analyzing-email-headers-for-phishing-investigation",
    "investigating-phishing-email-incident",
    "conducting-phishing-incident-response",
)
WINDOWS_LOGS = (
    "analyzing-windows-event-logs-in-splunk",
    "extracting-windows-event-logs-artifacts",
    "analyzing-security-logs-with-splunk",
)
MEMORY_FORENSICS = (
    "conducting-memory-forensics-with-volatility",
    "performing-memory-forensics-with-volatility3",
    "performing-memory-forensics-with-volatility3-plugins",
)
PACKET_ANALYSIS = (
    "analyzing-network-traffic-for-incidents",
    "analyzing-network-traffic-with-wireshark",
    "performing-network-traffic-analysis-with-tshark",
)


CASES = (
    RoutingCase("greeting", "conversation", "Hello", False),
    RoutingCase("thanks", "conversation", "Thanks for your help.", False),
    RoutingCase("test-message", "conversation", "This is a test", False),
    RoutingCase("connection-check", "conversation", "Are you there?", False),
    RoutingCase("arithmetic", "general", "What is 12 multiplied by 8?", False),
    RoutingCase(
        "project-fact",
        "statement",
        "My fictional project is called Blue Lantern.",
        False,
    ),
    RoutingCase("environment-fact", "statement", "Our lab uses Ubuntu servers.", False),
    RoutingCase(
        "ambiguous-test", "generic-overlap", "The test worked, thank you.", False
    ),
    RoutingCase(
        "ambiguous-network",
        "generic-overlap",
        "My network is called Blue Lantern.",
        False,
    ),
    RoutingCase(
        "security-definition",
        "general-security",
        "What does phishing mean? Answer in one sentence.",
        False,
    ),
    RoutingCase(
        "security-acronym", "general-security", "What does SIEM stand for?", False
    ),
    RoutingCase(
        "contextless-followup", "ambiguous-followup", "Continue with that.", False
    ),
    RoutingCase(
        "windows-failures",
        "specific-task",
        "Investigate Windows Event 4625 failed logons using Splunk.",
        True,
        (WINDOWS_LOGS,),
    ),
    RoutingCase(
        "short-technical",
        "short-task",
        "Investigate Event 4625.",
        True,
        (WINDOWS_LOGS,),
    ),
    RoutingCase(
        "phishing-headers",
        "specific-task",
        "Analyze this suspected phishing email's headers, including SPF, DKIM, DMARC and Received hops.",
        True,
        (PHISHING,),
    ),
    RoutingCase(
        "phishing-paraphrase",
        "paraphrase",
        "An employee received a fake password-reset email. Walk me through checking who really sent it.",
        True,
        (PHISHING,),
    ),
    RoutingCase(
        "kerberoasting-detection",
        "specific-task",
        "Help detect Kerberoasting from domain-controller events and unusual service-ticket requests.",
        True,
        (("detecting-kerberoasting-attacks",),),
    ),
    RoutingCase(
        "sigma-rule",
        "specific-task",
        "Write a Sigma detection rule for suspicious PowerShell encoded commands.",
        True,
        (("building-detection-rules-with-sigma",),),
    ),
    RoutingCase(
        "memory-analysis",
        "specific-task",
        "Investigate a Windows memory dump with Volatility 3 to find injected processes.",
        True,
        (MEMORY_FORENSICS,),
    ),
    RoutingCase(
        "packet-analysis",
        "specific-task",
        "Analyze a packet capture in Wireshark for suspicious outbound connections.",
        True,
        (PACKET_ANALYSIS,),
    ),
    RoutingCase(
        "yara-triage",
        "specific-task",
        "Triage these suspicious executable samples with YARA rules.",
        True,
        (
            (
                "performing-malware-triage-with-yara",
                "performing-threat-hunting-with-yara-rules",
            ),
        ),
    ),
    RoutingCase(
        "kubernetes-rbac",
        "specific-task",
        "Audit Kubernetes RBAC bindings for privilege escalation and overly broad roles.",
        True,
        (
            (
                "auditing-kubernetes-cluster-rbac",
                "auditing-kubernetes-rbac-privilege-escalation",
            ),
        ),
    ),
    RoutingCase(
        "test-prefix-with-task",
        "mixed-intent",
        "This is a test; investigate Windows Event 4625 failed logons using Splunk.",
        True,
        (WINDOWS_LOGS,),
    ),
    RoutingCase(
        "two-investigations",
        "multi-topic",
        "Investigate a phishing email's headers and separately analyze a suspicious packet capture with Wireshark.",
        True,
        (PHISHING, PACKET_ANALYSIS),
    ),
)


# Original cases remain a frozen baseline, not a tuning/validation mixture.
BASELINE_CASES = CASES
DEVELOPMENT_CASES = CASES + (
    RoutingCase(
        "troubleshooting-logins",
        "troubleshooting",
        "What is blocking legitimate logins?",
        True,
        (("detecting-anomalous-authentication-patterns",),),
    ),
    RoutingCase(
        "acronym-task",
        "short-acronym",
        "Audit AD for Kerberoasting.",
        True,
        (("detecting-kerberoasting-attacks",),),
    ),
    RoutingCase(
        "unrelated-security-title",
        "generic-overlap",
        "Suggest a title for my poem about network security.",
        False,
    ),
    RoutingCase(
        "negated-packet-task",
        "negation",
        "Do not analyze packet captures. Investigate this phishing email's headers and SPF failures.",
        True,
        (PHISHING,),
        PACKET_ANALYSIS,
        (PHISHING,),
    ),
    RoutingCase(
        "long-log-tail",
        "long-input",
        "heartbeat status=ok\n" * 600
        + "Investigate Windows Event 4625 failed logons using Splunk.",
        True,
        (WINDOWS_LOGS,),
    ),
)
# Never use this split for threshold tuning. Once consulted repeatedly, replace it.
VALIDATION_CASES = (
    RoutingCase("validation-greeting", "conversation", "Good evening!", False),
    RoutingCase(
        "validation-definition", "general-security", "Define Kerberoasting.", False
    ),
    RoutingCase(
        "validation-unrelated",
        "generic-overlap",
        "Translate the words network security into French.",
        False,
    ),
    RoutingCase(
        "validation-troubleshooting",
        "troubleshooting",
        "What is preventing successful logins?",
        True,
        (("detecting-anomalous-authentication-patterns",),),
    ),
    RoutingCase(
        "validation-short",
        "short-task",
        "Hunt Kerberoasting.",
        True,
        (("detecting-kerberoasting-attacks",),),
    ),
    RoutingCase(
        "validation-negation",
        "negation",
        "Ignore email analysis; examine this Wireshark capture for suspicious outbound traffic.",
        True,
        (PACKET_ANALYSIS,),
        PHISHING,
        (PACKET_ANALYSIS,),
    ),
    RoutingCase(
        "validation-multi",
        "multi-topic",
        "Check a forged email's sender headers, then inspect a suspicious Wireshark capture.",
        True,
        (PHISHING, PACKET_ANALYSIS),
    ),
    RoutingCase(
        "validation-long-head",
        "long-input",
        "Investigate Windows Event 4625 failed logons using Splunk.\n"
        + "heartbeat status=ok\n" * 600,
        True,
        (WINDOWS_LOGS,),
    ),
)
SPLITS = {
    "baseline": BASELINE_CASES,
    "development": DEVELOPMENT_CASES,
    "validation": VALIDATION_CASES,
}


def latency_summary(seconds):
    """Nearest-rank p95, with median; caller excludes warmup explicitly."""
    import math
    import statistics

    values = sorted(seconds)
    if not values:
        return {"samples": 0, "median_ms": None, "p95_ms": None}
    return {
        "samples": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1] * 1000,
    }


def fingerprint_model(model_path):
    """Hash local artifact bytes in bounded chunks; paths alone are insufficient."""
    root = Path(model_path).resolve()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        name = path.relative_to(root).as_posix()
        if any(part.startswith(".") for part in Path(name).parts):
            continue
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def benchmark_startup(repo, model_path, *, runs):
    """Fresh Python processes; filesystem page caches are not flushed."""
    if type(runs) is not int or runs < 1:
        raise ValueError("startup runs must be positive")
    samples = []
    for _ in range(runs):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--repo",
                str(repo),
                "--model-path",
                str(model_path),
                "--startup-probe",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
        )
        samples.append(json.loads(result.stdout)["startup_seconds"])
    return {
        **latency_summary(samples),
        "scope": "fresh_process_registry_and_router_initialization; excludes interpreter launch; OS caches uncontrolled",
    }


def benchmark_router(router, cases, *, limit=5, repeats=5, warmup=1):
    if any(
        type(value) is not int or value < minimum
        for value, minimum in ((repeats, 1), (warmup, 0), (limit, 1))
    ):
        raise ValueError("Invalid benchmark limits")
    cases = tuple(cases)
    if not cases:
        raise ValueError("Benchmark requires cases")
    rows = []
    for case in cases:
        for _ in range(warmup):
            router.route(case.prompt, limit=limit)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            router.route(case.prompt, limit=limit)
            times.append(time.perf_counter() - start)
        rows.append({"case_id": case.id, **latency_summary(times)})
    return {
        "warmup_per_case": warmup,
        "repeats_per_case": repeats,
        "cases": rows,
        "scope": "routing_only_excludes_instruction_loading_and_inference",
    }


def evaluate_router(
    router, cases=CASES, *, limit=5, include_text=False, measure_instructions=False
):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    cases = tuple(cases)
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("Evaluation requires nonempty cases with unique IDs")

    catalog = {skill.name for skill in router.registry.skills}
    for case in cases:
        if case.expect_skills != bool(case.required_groups):
            raise ValueError(f"Invalid expected groups for case {case.id}")
        for group in case.required_groups:
            if not group or not set(group).issubset(catalog):
                raise ValueError(f"Missing catalog skills for case {case.id}")

    for case in cases:
        relevant = (
            set().union(*map(set, case.required_groups))
            if case.required_groups
            else set()
        )
        if relevant.intersection(case.irrelevant_skills):
            raise ValueError(f"Conflicting judgments for {case.id}")
        if not set(case.irrelevant_skills).issubset(catalog):
            raise ValueError(f"Missing judged skills for {case.id}")
        for group in case.redundancy_groups:
            if len(set(group)) < 2 or not set(group).issubset(catalog):
                raise ValueError(f"Invalid redundancy group for {case.id}")
    rows = []
    for case in cases:
        # search() delegates to route(); share that actual activation path so
        # bypassed prompts do not encode and other queries encode only once.
        if hasattr(router, "route"):
            selected, decision = router.route(
                case.prompt, limit=limit, include_terms=include_text
            )
            diagnostics = decision["candidates"]
            reason = decision["reason"]
        else:
            selected = router.search(case.prompt, limit=limit)
            diagnostics = [
                item.diagnostic(include_terms=include_text)
                for item in router.score_candidates(case.prompt, limit=limit)
            ]
            reason = "external_router"
        names = {skill.name for _, skill in selected}
        groups_hit = sum(
            bool(names.intersection(group)) for group in case.required_groups
        )
        row = {
            "case_id": case.id,
            "category": case.category,
            "expected_activation": case.expect_skills,
            "actual_activation": bool(selected),
            "activation_correct": bool(selected) == case.expect_skills,
            "routing_reason": reason,
            "selected_skills": [skill.name for _, skill in selected],
            "required_groups": [list(group) for group in case.required_groups],
            "required_groups_hit": groups_hit,
            "required_groups_total": len(case.required_groups),
            "candidate_diagnostics": diagnostics,
            "query_plan": (
                decision.get("query_plan") if hasattr(router, "route") else None
            ),
        }
        relevant = (
            set().union(*map(set, case.required_groups))
            if case.required_groups
            else set()
        )
        # No-skill cases judge every selection irrelevant. Other unlabeled
        # skills remain unjudged, never implicitly counted as errors.
        labels = {
            name: (
                "irrelevant"
                if not case.expect_skills or name in case.irrelevant_skills
                else "relevant" if name in relevant else "unjudged"
            )
            for name in names
        }
        judged = sum(label != "unjudged" for label in labels.values())
        row.update(
            {
                "selection_judgments": labels,
                "judged_selected": judged,
                "relevant_selected": sum(
                    label == "relevant" for label in labels.values()
                ),
                "unjudged_selected": sum(
                    label == "unjudged" for label in labels.values()
                ),
                "redundancy_judgments_available": bool(case.redundancy_groups),
                "redundant_selected": (
                    len(
                        set().union(
                            *(
                                set(sorted(names.intersection(group))[1:])
                                for group in case.redundancy_groups
                            )
                        )
                    )
                    if case.redundancy_groups
                    else 0
                ),
            }
        )
        if measure_instructions:
            texts = [router.registry.load_instructions(skill) for _, skill in selected]
            row["selected_instruction_characters"] = sum(map(len, texts))
            # This is encoder-token cost, NOT inference-context token cost.
            tokenizer = router.model.tokenizer
            row["selected_instruction_embedding_tokens"] = sum(
                len(tokenizer.encode(text, add_special_tokens=False, truncation=False))
                for text in texts
            )
        if include_text:
            row["prompt"] = case.prompt
        rows.append(row)

    negatives = [row for row in rows if not row["expected_activation"]]
    positives = [row for row in rows if row["expected_activation"]]
    false_activations = sum(row["actual_activation"] for row in negatives)
    missed_activations = sum(not row["actual_activation"] for row in positives)
    total_groups = sum(row["required_groups_total"] for row in positives)
    hit_groups = sum(row["required_groups_hit"] for row in positives)
    catalog_metadata = sorted(
        (skill.name, skill.description, skill.domain)
        for skill in router.registry.skills
    )
    fingerprint = hashlib.sha256(
        json.dumps(catalog_metadata, ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    return {
        "evaluation_version": 4,
        "cases_sha256": hashlib.sha256(
            json.dumps([asdict(case) for case in cases], sort_keys=True).encode()
        ).hexdigest(),
        "catalog_sha256": fingerprint,
        "catalog_size": len(catalog),
        "configuration": {
            "selection_policy": (
                asdict(router.selection_policy)
                if hasattr(router, "selection_policy")
                else None
            ),
            "top_k": limit,
            "semantic_weight": router.semantic_weight,
            "name_boost": router.name_boost,
            "activation_policy": (
                asdict(router.activation_policy)
                if hasattr(router, "activation_policy")
                else None
            ),
            "query_plan_settings": (
                asdict(router.query_plan_settings)
                if hasattr(router, "query_plan_settings")
                else None
            ),
        },
        "summary": {
            "judged_selection_precision": (
                sum(row["relevant_selected"] for row in rows)
                / sum(row["judged_selected"] for row in rows)
                if sum(row["judged_selected"] for row in rows)
                else None
            ),
            "judgment_coverage": (
                sum(row["judged_selected"] for row in rows)
                / sum(len(row["selected_skills"]) for row in rows)
                if sum(len(row["selected_skills"]) for row in rows)
                else None
            ),
            "redundancy_judged_cases": sum(
                row["redundancy_judgments_available"] for row in rows
            ),
            "redundant_selected": sum(row["redundant_selected"] for row in rows),
            "cases": len(rows),
            "negative_cases": len(negatives),
            "positive_cases": len(positives),
            "false_activations": false_activations,
            "false_activation_rate": (
                false_activations / len(negatives) if negatives else None
            ),
            "missed_activations": missed_activations,
            "missed_activation_rate": (
                missed_activations / len(positives) if positives else None
            ),
            "expected_group_coverage": (
                hit_groups / total_groups if total_groups else None
            ),
            "positive_cases_with_all_groups": sum(
                row["required_groups_hit"] == row["required_groups_total"]
                for row in positives
            ),
        },
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--semantic-weight", type=float, default=0.60)
    parser.add_argument("--name-boost", type=float, default=3.0)
    parser.add_argument("--legacy-routing", action="store_true")
    parser.add_argument("--legacy-selection", action="store_true")
    parser.add_argument("--redundancy-similarity", type=float, default=0.70)
    parser.add_argument("--relative-score-floor", type=float, default=0.80)
    parser.add_argument("--novelty-weight", type=float, default=0.25)
    parser.add_argument("--min-semantic-score", type=float, default=0.36)
    parser.add_argument("--min-combined-score", type=float, default=0.45)
    parser.add_argument("--semantic-only-score", type=float, default=0.65)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Include synthetic prompts and matching terms",
    )
    parser.add_argument("--split", choices=tuple(SPLITS), default="baseline")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--startup-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure-instructions", action="store_true")
    add_query_plan_arguments(parser)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0 or args.cold_runs < 1:
        parser.error("repeats must be positive and warmup nonnegative")
    if args.top <= 0:
        parser.error("--top must be positive")
    if not 0 <= args.semantic_weight <= 1:
        parser.error("--semantic-weight must be between 0 and 1")
    if not 0 <= args.name_boost < float("inf"):
        parser.error("--name-boost must be finite and nonnegative")
    try:
        query_settings = query_plan_settings_from_arguments(args)
        selection = RouterSelectionPolicy(
            not args.legacy_selection,
            args.redundancy_similarity,
            args.relative_score_floor,
            args.novelty_weight,
        )
        policy = RouterPolicy(
            enabled=not args.legacy_routing,
            min_semantic_score=args.min_semantic_score,
            min_combined_score=args.min_combined_score,
            semantic_only_score=args.semantic_only_score,
        )
    except ValueError as exc:
        parser.error(str(exc))
    requested = set(args.case_ids or ())
    if requested - {case.id for case in SPLITS[args.split]}:
        parser.error("Unknown --case ID")
    cases = tuple(
        case for case in SPLITS[args.split] if not requested or case.id in requested
    )

    # Keep stdout a valid JSON report. Existing startup notices go to stderr.
    with redirect_stdout(sys.stderr):
        start = time.perf_counter()
        registry = SkillRegistry(args.repo)
        router = SkillRouter(
            registry,
            model_path=args.model_path,
            semantic_weight=args.semantic_weight,
            name_boost=args.name_boost,
            activation_policy=policy,
            selection_policy=selection,
            query_plan_settings=query_settings,
        )
        startup_seconds = time.perf_counter() - start
        if args.startup_probe:
            # The stdout redirect remains active here; write the probe to
            # original stdout so the parent receives exactly one JSON object.
            sys.__stdout__.write(
                json.dumps({"startup_seconds": startup_seconds}) + "\n"
            )
            return
        report = evaluate_router(
            router,
            cases,
            limit=args.top,
            include_text=args.show_text,
            measure_instructions=args.measure_instructions,
        )
        report["split"] = args.split
        report["model_sha256"] = fingerprint_model(args.model_path)
        tokenizer = router.model.tokenizer
        report["encoder"] = {
            "max_seq_length": router.model.max_seq_length,
            "tokenizer_class": type(tokenizer).__name__,
            "model_max_length": tokenizer.model_max_length,
            "truncation_side": tokenizer.truncation_side,
            "normalize_embeddings": True,
            "device": str(router.model.device),
        }
        report["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("sentence-transformers", "transformers", "torch")
            },
        }
        report["startup"] = {
            "seconds": startup_seconds,
            "scope": "registry_and_router_initialization; OS filesystem cache uncontrolled",
        }
        if args.benchmark:
            report["cold_startup"] = benchmark_startup(
                args.repo, args.model_path, runs=args.cold_runs
            )
            report["benchmark"] = benchmark_router(
                router, cases, limit=args.top, repeats=args.repeats, warmup=args.warmup
            )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
