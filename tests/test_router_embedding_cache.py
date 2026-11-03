"""Deterministic cache correctness tests; timing is never a performance claim."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analyst"))
import router_embedding_cache as cache_module
from router_embedding_cache import (
    CatalogEmbeddingCache, EmbeddingCacheSettings, artifact_fingerprint,
    canonical_json, embedding_contract, load_embedding_cache_settings,
    read_cache, validate_matrix, write_cache,
)
from orchestrator import Skill, SkillRouter, tokenize
from router_policy import RouterPolicy, RouterSelectionPolicy
from router_query_plan import QueryPlanSettings


NAMES = ["windows-events", "phishing-headers"]
TEXTS = ["Investigate Windows events", "Analyze phishing headers"]
VECTORS = np.array([[1.0, 0.0], [0.6, 0.8]], dtype="<f8")


class FakeModel:
    max_seq_length = 256
    device = "cpu"
    dtype = "torch.float32"
    truncate_dim = None
    prompts = {}
    default_prompt_name = None

    def __init__(self):
        self.tokenizer = SimpleNamespace(
            model_max_length=512, truncation_side="right", padding_side="right",
            special_tokens_map={}, init_kwargs={},
        )
        self._modules = {}
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 2

    def encode(self, text, **kwargs):
        self.calls.append(text)
        if isinstance(text, list):
            return VECTORS[:len(text)].copy()
        return np.array([0.6, 0.8])


def _process_cache(model_root, directory, start, queue, building=None, release=None):
    """Fresh processes exercise real locks/files; the encoder itself is fake."""
    import time
    model = FakeModel()
    start.wait(20)

    def build():
        fd = os.open(Path(directory) / "builds.log", os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.write(fd, b"build\n")
        finally:
            os.close(fd)
        if building is not None:
            building.set()
        if release is not None:
            if not release.wait(20):
                raise RuntimeError("Test release timed out")
        else:
            time.sleep(0.15)
        return VECTORS.copy()

    try:
        matrix, report = CatalogEmbeddingCache(EmbeddingCacheSettings(
            True, Path(directory), 10
        )).load_or_build(model=model, model_path=model_root, names=NAMES, texts=TEXTS, build=build)
        queue.put((report["status"], matrix.tobytes()))
    except Exception as exc:
        queue.put(("error", repr(exc)))
        raise


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        (self.model_root / "model.safetensors").write_bytes(b"local-model-v1")
        (self.model_root / "tokenizer.json").write_text("{}")
        self.directory = self.root / "cache"
        self.settings = EmbeddingCacheSettings(True, self.directory, 5)
        self.model = FakeModel()

    def call(self, *, settings=None, model=None, names=None, texts=None, build=None):
        builder = build if build is not None else Mock(return_value=VECTORS.copy())
        result = CatalogEmbeddingCache(settings or self.settings).load_or_build(
            model=model or self.model, model_path=self.model_root,
            names=NAMES if names is None else names,
            texts=TEXTS if texts is None else texts, build=builder,
        )
        return result[0], result[1], builder

    def path(self):
        return next(self.directory.glob("*.npz"))

    def rewrite(self, *, manifest_change=None, matrix=None, manifest_array=None):
        path = self.path()
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["manifest"].tobytes())
            values = archive["matrix"].copy()
        if manifest_change:
            manifest_change(metadata)
        if matrix is not None:
            values = matrix
        if manifest_array is None:
            manifest_array = np.frombuffer(canonical_json(metadata), dtype="u1")
        with path.open("wb") as stream:
            np.savez(stream, manifest=manifest_array, matrix=values)

    def test_build_then_hit_is_bitwise_equal_readonly_and_skips_encode(self):
        first, report, build = self.call()
        self.assertEqual(report["status"], "built")
        build.assert_called_once_with()
        second, report, build = self.call(build=Mock(side_effect=AssertionError("encoded on hit")))
        self.assertEqual(report["status"], "hit")
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.tobytes(), VECTORS.tobytes())
        self.assertFalse(second.flags.writeable)
        build.assert_not_called()

    def test_disabled_does_not_touch_storage_or_fingerprint(self):
        settings = EmbeddingCacheSettings(False, self.directory)
        with patch.object(cache_module, "embedding_contract", side_effect=AssertionError):
            _, report, build = self.call(settings=settings)
        self.assertEqual(report["status"], "disabled")
        build.assert_called_once_with()
        self.assertFalse(self.directory.exists())

    def test_empty_catalog_never_encodes_or_writes(self):
        matrix, report, build = self.call(names=[], texts=[])
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(report["status"], "empty_catalog")
        build.assert_not_called()
        self.assertFalse(self.directory.exists())

    def test_names_and_texts_must_be_one_to_one(self):
        for names, texts in ((["x", "x"], TEXTS), (["x"], TEXTS)):
            with self.assertRaises(ValueError):
                self.call(names=names, texts=texts)

    def test_local_artifact_bytes_invalidate_even_with_same_size_and_mtime(self):
        _, initial, _ = self.call()
        path = self.model_root / "model.safetensors"
        old_stat = path.stat()
        path.write_bytes(b"local-model-v2")
        os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        _, current, build = self.call()
        self.assertNotEqual(initial["key"], current["key"])
        self.assertEqual(current["status"], "built")
        build.assert_called_once_with()

    def test_tokenizer_artifact_and_hidden_config_invalidate(self):
        before = artifact_fingerprint(self.model_root)
        (self.model_root / "tokenizer.json").write_text('{"truncation_side":"left"}')
        after = artifact_fingerprint(self.model_root)
        self.assertNotEqual(before, after)
        (self.model_root / ".settings").write_text("new setting")
        self.assertNotEqual(after, artifact_fingerprint(self.model_root))

    def test_effective_tokenizer_model_prompts_and_metadata_invalidate(self):
        _, initial, _ = self.call()
        mutations = (
            lambda model: setattr(model.tokenizer, "model_max_length", 128),
            lambda model: setattr(model.tokenizer, "truncation_side", "left"),
            lambda model: setattr(model.tokenizer, "padding_side", "left"),
            lambda model: setattr(model, "max_seq_length", 128),
            lambda model: setattr(model, "prompts", {"catalog": "Find: "}),
            lambda model: setattr(model, "default_prompt_name", "catalog"),
            lambda model: setattr(model, "dtype", "torch.float64"),
            lambda model: setattr(model, "_modules", {"pool": SimpleNamespace(
                get_config_dict=lambda: {"pooling_mode_mean_tokens": False})}),
        )
        for mutate in mutations:
            model = FakeModel()
            mutate(model)
            _, report, _ = self.call(model=model)
            self.assertNotEqual(initial["key"], report["key"])
            self.assertEqual(report["status"], "built")
        _, report, _ = self.call(texts=[TEXTS[0] + " changed", TEXTS[1]])
        self.assertNotEqual(initial["key"], report["key"])
        _, report, _ = self.call(names=list(reversed(NAMES)), texts=list(reversed(TEXTS)),
                                 build=Mock(return_value=VECTORS[::-1]))
        self.assertNotEqual(initial["key"], report["key"])

    def test_preprocessing_encode_settings_and_package_versions_invalidate(self):
        _, initial, _ = self.call()
        with patch.object(cache_module, "PREPROCESSING_VERSION", "next"):
            _, report, _ = self.call()
            self.assertNotEqual(initial["key"], report["key"])
        with patch.dict(cache_module.ENCODE_SETTINGS, batch_size=16):
            _, report, _ = self.call()
            self.assertNotEqual(initial["key"], report["key"])
        with patch.dict(cache_module.ENCODE_SETTINGS, normalize_embeddings=False):
            _, report, _ = self.call()
            self.assertNotEqual(initial["key"], report["key"])
        with patch.object(cache_module, "version", return_value="different"):
            _, report, _ = self.call()
            self.assertNotEqual(initial["key"], report["key"])

    def test_inference_environment_does_not_invalidate(self):
        _, initial, _ = self.call()
        with patch.dict(os.environ, MODEL_NAME="replacement-llm", LLM_BASE_URL="unused"):
            _, report, build = self.call()
        self.assertEqual(initial["key"], report["key"])
        self.assertEqual(report["status"], "hit")
        build.assert_not_called()

    def test_model_dimension_change_builds_a_separate_valid_matrix(self):
        _, original, _ = self.call()
        model = FakeModel()
        model.get_sentence_embedding_dimension = lambda: 3
        matrix = np.column_stack((VECTORS, np.zeros(2)))
        actual, report, _ = self.call(model=model, build=Mock(return_value=matrix))
        self.assertEqual(actual.shape, (2, 3))
        self.assertEqual(report["status"], "built")
        self.assertNotEqual(original["key"], report["key"])

    def test_cache_does_not_include_request_text(self):
        self.call()
        with np.load(self.path(), allow_pickle=False) as archive:
            manifest = json.loads(archive["manifest"].tobytes())
            self.assertEqual(set(manifest), {
                "format", "key", "names", "rows", "dimension", "dtype", "matrix_sha256"
            })
            self.assertEqual(manifest["names"], NAMES)

    def test_corrupt_and_truncated_files_are_rebuilt(self):
        for data in (b"", b"not a zip", b"PK\x03\x04", b"[]"):
            self.call()
            self.path().write_bytes(data)
            _, report, build = self.call()
            self.assertEqual(report["status"], "built")
            self.assertEqual(report["lookup_reason"], "invalid_or_unreadable")
            build.assert_called_once_with()

    def test_invalid_manifests_names_shapes_dtypes_and_values_are_rebuilt(self):
        mutations = (
            {"manifest_change": lambda data: data.update(format=999)},
            {"manifest_change": lambda data: data.update(format=True)},
            {"manifest_change": lambda data: data.update(key="wrong")},
            {"manifest_change": lambda data: data.update(names=NAMES[::-1])},
            {"manifest_change": lambda data: data.update(dimension=3)},
            {"manifest_change": lambda data: data.update(matrix_sha256="0" * 64)},
            {"manifest_change": lambda data: data.update(extra=True)},
            {"matrix": VECTORS[:, :1]},
            {"matrix": VECTORS[:1]},
            {"matrix": VECTORS.astype(np.float32)},
            {"matrix": np.array([[np.nan, 0.0], [0.6, 0.8]])},
            {"matrix": np.array([[np.inf, 0.0], [0.6, 0.8]])},
            {"matrix": np.zeros((2, 2))},
            {"matrix": VECTORS * 2},
            {"matrix": np.array([["a", "b"], ["c", "d"]], dtype=object)},
            {"manifest_array": np.array([{}], dtype=object)},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.call()
                self.rewrite(**mutation)
                _, report, build = self.call()
                self.assertEqual(report["status"], "built")
                build.assert_called_once_with()

    def test_bad_numeric_rows_rejected_even_with_matching_checksum(self):
        self.call()
        for matrix in (VECTORS * 2, np.zeros((2, 2)),
                       np.array([[float("nan"), 0], [0.6, 0.8]])):
            self.rewrite(matrix=matrix, manifest_change=lambda data: data.update(
                matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest()))
            _, report, _ = self.call()
            self.assertEqual(report["status"], "built")

    def test_extra_archive_members_and_compressed_members_rejected(self):
        import zipfile
        self.call()
        with zipfile.ZipFile(self.path(), "a") as archive:
            archive.writestr("extra", b"data")
        _, report, _ = self.call()
        self.assertEqual(report["status"], "built")
        with np.load(self.path(), allow_pickle=False) as archive:
            manifest, matrix = archive["manifest"], archive["matrix"]
        with self.path().open("wb") as stream:
            np.savez_compressed(stream, manifest=manifest, matrix=matrix)
        _, report, _ = self.call()
        self.assertEqual(report["status"], "built")

    def test_oversized_archive_rejected_before_numpy_loading(self):
        self.call()
        with patch.object(cache_module, "MAX_ARCHIVE_BYTES", 1):
            with patch.object(np, "load", side_effect=AssertionError("loaded oversized cache")):
                _, report, _ = self.call()
        self.assertEqual(report["status"], "built")

    def test_abandoned_temporary_archive_is_ignored(self):
        self.directory.mkdir()
        orphan = self.directory / ".abandoned.tmp"
        orphan.write_bytes(b"partial archive")
        _, report, _ = self.call()
        self.assertEqual(report["status"], "built")
        self.assertEqual(orphan.read_bytes(), b"partial archive")

    def test_oversized_headers_rejected_before_numpy_load(self):
        stream = io.BytesIO()
        np.lib.format.write_array_header_1_0(stream, {
            "descr": "<f8", "fortran_order": False, "shape": (10**10, 10**10),
        })
        with patch.object(np, "load", side_effect=AssertionError("allocated")):
            with self.assertRaises(ValueError):
                cache_module._array_from_bytes(stream.getvalue(), (2, 2), "<f8")

    def test_invalid_fresh_vectors_fail_without_publication(self):
        for matrix in (VECTORS[:, :1], VECTORS[:1], VECTORS * 2,
                       np.array([[float("nan"), 0], [0.6, 0.8]])):
            with self.assertRaises(ValueError):
                self.call(build=Mock(return_value=matrix))
            self.assertFalse(list(self.directory.glob("*.npz")))
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            self.call(build=Mock(side_effect=RuntimeError("model failure")))

    def test_fingerprint_failure_builds_privately(self):
        with patch.object(cache_module, "artifact_fingerprint", side_effect=PermissionError):
            _, report, build = self.call()
        self.assertEqual(report["status"], "fingerprint_unavailable")
        build.assert_called_once_with()
        self.assertFalse(self.directory.exists())

    def test_unwritable_storage_builds_privately(self):
        self.directory.write_text("a file instead of a directory")
        _, report, build = self.call()
        self.assertEqual(report["status"], "storage_unavailable")
        build.assert_called_once_with()

    def test_failed_atomic_replace_preserves_previous_file_and_cleans_temp(self):
        matrix, report, _ = self.call()
        path = self.path()
        original = path.read_bytes()
        with patch.object(cache_module.os, "replace", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                write_cache(path, report["key"], NAMES, matrix)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(self.directory.glob("*.tmp")))
        self.assertFalse(list(self.directory.glob(".*.tmp")))

    def test_readers_see_complete_old_or_complete_new_archive(self):
        original, report, _ = self.call()
        replacement = VECTORS[::-1].copy()
        path = self.path()
        replace = os.replace

        def publish(source, destination):
            visible = read_cache(path, report["key"], NAMES, 2)
            self.assertEqual(visible.tobytes(), original.tobytes())
            # The complete temporary archive is valid before publication too.
            ready = read_cache(Path(source), report["key"], NAMES, 2)
            self.assertEqual(ready.tobytes(), replacement.tobytes())
            return replace(source, destination)

        with patch.object(cache_module.os, "replace", side_effect=publish):
            write_cache(path, report["key"], NAMES, replacement)
        visible = read_cache(path, report["key"], NAMES, 2)
        self.assertEqual(visible.tobytes(), replacement.tobytes())

    def test_publication_failure_does_not_repeat_encoding(self):
        with patch.object(cache_module, "write_cache", side_effect=OSError("simulated")):
            _, report, build = self.call()
        self.assertEqual(report["status"], "write_failed")
        build.assert_called_once_with()

    def test_lock_timeout_builds_privately_without_publishing(self):
        _, report, _ = self.call()
        path = self.path()
        path.unlink()
        lock_fd = os.open(path.with_suffix(".lock"), os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _, result, build = self.call(settings=EmbeddingCacheSettings(True, self.directory, 0.05))
        finally:
            os.close(lock_fd)
        self.assertEqual(result["key"], report["key"])
        self.assertEqual(result["status"], "lock_timeout")
        build.assert_called_once_with()
        self.assertFalse(path.exists())

    def test_concurrent_fresh_processes_build_once(self):
        self.directory.mkdir()
        context = multiprocessing.get_context("spawn")
        start, queue = context.Event(), context.Queue()
        processes = [context.Process(target=_process_cache, args=(
            str(self.model_root), str(self.directory), start, queue
        )) for _ in range(4)]
        try:
            for process in processes:
                process.start()
            start.set()
            results = [queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sum(status == "built" for status, _ in results), 1)
            self.assertTrue(all(status in {"built", "hit", "hit_after_wait"} for status, _ in results))
            self.assertTrue(all(data == VECTORS.tobytes() for _, data in results))
            self.assertEqual((self.directory / "builds.log").read_text().splitlines(), ["build"])
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            queue.close()

    def test_terminated_builder_releases_lock_and_next_start_rebuilds(self):
        self.directory.mkdir()
        context = multiprocessing.get_context("spawn")
        start, building, release = context.Event(), context.Event(), context.Event()
        queue = context.Queue()
        process = context.Process(target=_process_cache, args=(
            str(self.model_root), str(self.directory), start, queue, building, release
        ))
        process.start()
        try:
            start.set()
            self.assertTrue(building.wait(20))
            process.terminate()
            process.join(10)
            _, report, _ = self.call()
            self.assertEqual(report["status"], "built")
        finally:
            if process.is_alive():
                process.terminate()
            process.join(5)
            queue.close()

    def test_settings_validation(self):
        self.assertTrue(load_embedding_cache_settings({}).enabled)
        self.assertFalse(load_embedding_cache_settings({"ROUTER_EMBEDDING_CACHE_ENABLED": "false"}).enabled)
        for enabled in (1, "yes", None):
            with self.assertRaises(ValueError):
                EmbeddingCacheSettings(enabled=enabled)
        for timeout in (0, -1, True, float("nan"), 601):
            with self.assertRaises(ValueError):
                EmbeddingCacheSettings(lock_timeout_seconds=timeout)
        with self.assertRaises(ValueError):
            load_embedding_cache_settings({"ROUTER_EMBEDDING_CACHE_ENABLED": "yes"})
        with self.assertRaises(ValueError):
            EmbeddingCacheSettings(directory=Path("relative"))

    def test_router_cached_and_uncached_results_and_call_counts(self):
        skills = [Skill(name, text, "security", name, tuple(tokenize(f"{name} {text}")))
                  for name, text in zip(NAMES, TEXTS)]
        registry = SimpleNamespace(skills=skills, load_instructions=Mock())

        def router(enabled):
            with patch.object(SkillRouter, "_load_embedding_model", return_value=FakeModel()):
                with redirect_stdout(io.StringIO()):
                    return SkillRouter(
                        registry, model_path=self.model_root, semantic_weight=.6, name_boost=3.,
                        activation_policy=RouterPolicy(), selection_policy=RouterSelectionPolicy(),
                        query_plan_settings=QueryPlanSettings(enabled=False),
                        embedding_cache_settings=EmbeddingCacheSettings(enabled, self.directory),
                    )

        uncached, built, hit = router(False), router(True), router(True)
        self.assertEqual([r.embedding_calls["catalog"] for r in (uncached, built, hit)], [1, 1, 0])
        self.assertEqual(uncached.embedding_matrix.tobytes(), hit.embedding_matrix.tobytes())
        self.assertEqual(uncached.skill_embeddings, hit.skill_embeddings)
        for prompt in ("Hello", "This is a test", "Continue with that.",
                       "Investigate Windows events", "Analyze phishing headers",
                       "Do not analyze phishing; investigate Windows events.",
                       "Investigate Windows events then analyze phishing headers."):
            results = [r.route(prompt) for r in (uncached, built, hit)]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
        self.assertEqual(uncached.embedding_calls["query"], hit.embedding_calls["query"])
        self.assertEqual(hit.embedding_calls["catalog"], 0)
        registry.load_instructions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
