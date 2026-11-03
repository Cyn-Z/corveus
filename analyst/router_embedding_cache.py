"""Local catalog vectors only. No prompts, inference state, or pickle loading.

Linux flock coordinates builders; atomic replacement protects readers. Model
artifacts and the registry must remain immutable during router initialization.
Cache storage is operator-controlled; checksums detect damage, not forgery.
"""
from __future__ import annotations

import errno
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1
PREPROCESSING_VERSION = "skill-name-domain-description-v1"
ENCODE_SETTINGS = {
    "batch_size": 32,
    "convert_to_numpy": True,
    "normalize_embeddings": True,
    "show_progress_bar": False,
}
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_MATRIX_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_MANIFEST_BYTES + MAX_MATRIX_BYTES + 65536
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingCacheSettings:
    # Direct library/test callers remain opt-in; application loading is explicit.
    enabled: bool = False
    directory: Path = Path("/app/data/router-embeddings")
    lock_timeout_seconds: float = 120.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("ROUTER_EMBEDDING_CACHE_ENABLED must be boolean")
        directory = Path(self.directory)
        if not directory.is_absolute():
            raise ValueError("ROUTER_EMBEDDING_CACHE_DIR must be absolute")
        object.__setattr__(self, "directory", directory)
        if (type(self.lock_timeout_seconds) not in (int, float)
                or not math.isfinite(self.lock_timeout_seconds)
                or not 0 < self.lock_timeout_seconds <= 600):
            raise ValueError("Cache lock timeout must be finite and in (0, 600]")


def load_embedding_cache_settings(environ=None):
    env = os.environ if environ is None else environ
    enabled = env.get("ROUTER_EMBEDDING_CACHE_ENABLED", "true").strip().lower()
    if enabled not in {"true", "false"}:
        raise ValueError("ROUTER_EMBEDDING_CACHE_ENABLED must be true or false")
    return EmbeddingCacheSettings(
        enabled=enabled == "true",
        directory=Path(env.get(
            "ROUTER_EMBEDDING_CACHE_DIR", "/app/data/router-embeddings"
        )),
    )


def add_embedding_cache_arguments(parser):
    parser.add_argument("--embedding-cache-dir", help="Absolute local cache directory; enables caching")
    parser.add_argument("--disable-embedding-cache", action="store_true")


def embedding_cache_settings_from_arguments(args):
    return EmbeddingCacheSettings(
        enabled=bool(args.embedding_cache_dir) and not args.disable_embedding_cache,
        directory=Path(args.embedding_cache_dir or "/app/data/router-embeddings"),
    )


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def artifact_fingerprint(model_path):
    """Hash every local file, including hidden config; do not memoize by mtime."""
    root = Path(model_path).resolve()
    digest = hashlib.sha256()
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    if not paths:
        raise ValueError("Embedding model directory has no artifacts")
    for path in paths:
        digest.update(canonical_json(path.relative_to(root).as_posix()) + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _json_value(value):
    """Explicit conversion for token/config objects; never serialize code."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def embedding_contract(model, model_path, names, texts):
    """No inference model, routing weights, context settings, or user data."""
    tokenizer = model.tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    modules = []
    for name, module in getattr(model, "_modules", {}).items():
        config = getattr(module, "get_config_dict", None)
        modules.append({
            "name": name,
            "class": f"{type(module).__module__}.{type(module).__qualname__}",
            "config": _json_value(config()) if callable(config) else None,
        })
    packages = {}
    for name in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    contract = {
        "format": FORMAT_VERSION,
        "preprocessing": PREPROCESSING_VERSION,
        "artifacts_sha256": artifact_fingerprint(model_path),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "device": str(model.device),
        "dtype": str(getattr(model, "dtype", None)),
        "truncate_dim": getattr(model, "truncate_dim", None),
        "similarity_fn_name": _json_value(getattr(model, "similarity_fn_name", None)),
        "prompts": _json_value(getattr(model, "prompts", {})),
        "default_prompt_name": getattr(model, "default_prompt_name", None),
        "modules": modules,
        "tokenizer": {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "model_max_length": tokenizer.model_max_length,
            "truncation_side": getattr(tokenizer, "truncation_side", None),
            "padding_side": getattr(tokenizer, "padding_side", None),
            "special_tokens_map": _json_value(getattr(tokenizer, "special_tokens_map", {})),
            "init_kwargs": _json_value(getattr(tokenizer, "init_kwargs", {})),
            "backend": backend.to_str() if backend is not None else None,
        },
        "encode": ENCODE_SETTINGS,
        "matrix_dtype": "<f8",
        "packages": packages,
        "catalog": list(zip(names, texts)),
    }
    dimension = contract["dimension"]
    if type(dimension) is not int or not 1 <= dimension <= MAX_MATRIX_BYTES // 8:
        raise ValueError("Invalid embedding model dimension")
    return contract


def validate_matrix(value, rows, dimension=None):
    """Same validation for freshly built and loaded vectors; never renormalize."""
    original = np.asarray(value)
    if original.dtype.kind != "f":
        raise ValueError("Embedding matrix must contain floating-point values")
    if original.ndim != 2 or original.shape[0] != rows:
        raise ValueError("Embedding matrix row count/shape mismatch")
    if not 1 <= original.shape[1] <= MAX_MATRIX_BYTES // 8:
        raise ValueError("Invalid embedding dimension")
    if dimension is not None and original.shape[1] != dimension:
        raise ValueError("Embedding matrix dimension mismatch")
    if original.size * 8 > MAX_MATRIX_BYTES:
        raise ValueError("Embedding matrix exceeds the size limit")
    matrix = np.array(original, dtype="<f8", order="C", copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix is oversized or contains nonfinite values")
    if rows and not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=0, atol=1e-4):
        raise ValueError("Embedding rows must be nonzero normalized vectors")
    matrix.setflags(write=False)
    return matrix


def _array_from_bytes(data, shape, dtype):
    """Check NPY headers and payload length before any array allocation."""
    stream = io.BytesIO(data)
    npy_version = np.lib.format.read_magic(stream)
    if npy_version == (1, 0):
        header = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
    elif npy_version == (2, 0):
        header = np.lib.format.read_array_header_2_0(stream, max_header_size=10000)
    else:
        raise ValueError("Unsupported NPY version")
    found_shape, fortran, found_dtype = header
    if found_shape != shape or fortran or found_dtype.str != np.dtype(dtype).str:
        raise ValueError("Unexpected NPY shape, order, or dtype")
    expected_size = math.prod(shape) * np.dtype(dtype).itemsize
    if len(data) - stream.tell() != expected_size:
        raise ValueError("Unexpected NPY payload length")
    # allow_pickle=False is also explicit even after strict numeric header checks.
    stream.seek(0)
    return np.load(stream, allow_pickle=False, max_header_size=10000)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key")
        result[key] = value
    return result


def read_cache(path, key, names, dimension):
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Invalid cache file size or type")
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist()
            if sorted(item.filename for item in entries) != ["manifest.npy", "matrix.npy"]:
                raise ValueError("Unexpected cache archive members")
            for item in entries:
                bound = MAX_MANIFEST_BYTES if item.filename == "manifest.npy" else MAX_MATRIX_BYTES
                if (item.compress_type != zipfile.ZIP_STORED or item.flag_bits & 1
                        or item.file_size > bound + 10000):
                    raise ValueError("Invalid cache member size or compression")
            raw_manifest = archive.read("manifest.npy")
            # Manifest length is discovered without trusting the shape in its header.
            header_stream = io.BytesIO(raw_manifest)
            npy_version = np.lib.format.read_magic(header_stream)
            readers = {(1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0}
            if npy_version not in readers:
                raise ValueError("Unsupported manifest NPY version")
            shape, _, _ = readers[npy_version](header_stream, max_header_size=10000)
            if (len(shape) != 1 or type(shape[0]) is not int
                    or not 0 < shape[0] <= MAX_MANIFEST_BYTES):
                raise ValueError("Invalid manifest length")
            manifest_array = _array_from_bytes(raw_manifest, shape, "u1")
            manifest = json.loads(manifest_array.tobytes().decode("ascii"),
                                  object_pairs_hook=_strict_object)
            if not isinstance(manifest, dict) or set(manifest) != {
                "format", "key", "names", "rows", "dimension", "dtype", "matrix_sha256"
            }:
                raise ValueError("Invalid cache manifest")
            if (type(manifest["format"]) is not int or manifest["format"] != FORMAT_VERSION
                    or manifest["key"] != key or manifest["names"] != list(names)
                    or type(manifest["rows"]) is not int or manifest["rows"] != len(names)
                    or type(manifest["dimension"]) is not int
                    or manifest["dimension"] != dimension or manifest["dtype"] != "<f8"):
                raise ValueError("Stale cache manifest")
            matrix = _array_from_bytes(archive.read("matrix.npy"), (len(names), dimension), "<f8")
            if hashlib.sha256(matrix.tobytes(order="C")).hexdigest() != manifest["matrix_sha256"]:
                raise ValueError("Cache matrix checksum mismatch")
            return validate_matrix(matrix, len(names), dimension)


def write_cache(path, key, names, matrix):
    manifest = canonical_json({
        "format": FORMAT_VERSION, "key": key, "names": list(names),
        "rows": len(names), "dimension": matrix.shape[1], "dtype": "<f8",
        "matrix_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
    })
    if len(manifest) > MAX_MANIFEST_BYTES:
        raise ValueError("Cache manifest exceeds the size limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{key}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            np.savez(stream, manifest=np.frombuffer(manifest, dtype="u1"), matrix=matrix)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CatalogEmbeddingCache:
    def __init__(self, settings):
        self.settings = settings

    def load_or_build(self, *, model, model_path, names, texts, build):
        started = time.perf_counter()
        report = {"status": "disabled", "key": None, "lookup_reason": None,
                  "fingerprint_seconds": 0.0, "lock_wait_seconds": 0.0}

        def finish(matrix, status):
            report["status"] = status
            report["seconds"] = time.perf_counter() - started
            return matrix, report

        if len(names) != len(texts) or len(set(names)) != len(names):
            raise ValueError("Catalog names and texts must be one-to-one")
        if not names:
            # No disk/model contract needed; route() already bypasses empty catalogs.
            matrix = np.empty((0, 0), dtype="<f8")
            matrix.setflags(write=False)
            return finish(matrix, "empty_catalog")
        if not self.settings.enabled:
            dimension_fn = getattr(model, "get_sentence_embedding_dimension", None)
            dimension = dimension_fn() if callable(dimension_fn) else None
            return finish(validate_matrix(build(), len(names), dimension), "disabled")

        before_fingerprint = time.perf_counter()
        try:
            contract = embedding_contract(model, model_path, names, texts)
            key = hashlib.sha256(canonical_json(contract)).hexdigest()
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            LOGGER.warning("Catalog cache unavailable during fingerprint: %s", type(exc).__name__)
            report["fingerprint_seconds"] = time.perf_counter() - before_fingerprint
            dimension_fn = getattr(model, "get_sentence_embedding_dimension", None)
            dimension = dimension_fn() if callable(dimension_fn) else None
            return finish(validate_matrix(build(), len(names), dimension), "fingerprint_unavailable")
        report["fingerprint_seconds"] = time.perf_counter() - before_fingerprint
        report["key"] = key
        dimension = contract["dimension"]
        path = self.settings.directory / f"{key}.npz"

        def build_validated():
            return validate_matrix(build(), len(names), dimension)

        def lookup():
            try:
                return read_cache(path, key, names, dimension)
            except FileNotFoundError:
                report["lookup_reason"] = "missing"
            except (OSError, ValueError, TypeError, KeyError, EOFError,
                    UnicodeError, zipfile.BadZipFile, OverflowError, RecursionError):
                report["lookup_reason"] = "invalid_or_unreadable"
            return None

        matrix = lookup()
        if matrix is not None:
            return finish(matrix, "hit")
        # Catch only cache I/O errors, never encode/model errors.
        try:
            self.settings.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock_fd = os.open(path.with_suffix(".lock"),
                              os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        except OSError as exc:
            LOGGER.warning("Catalog cache storage unavailable: %s", type(exc).__name__)
            return finish(build_validated(), "storage_unavailable")
        acquired = False
        wait_started = time.monotonic()
        lock_error = False
        try:
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EACCES):
                        lock_error = True
                        break
                    elapsed = time.monotonic() - wait_started
                    if elapsed >= self.settings.lock_timeout_seconds:
                        break
                    time.sleep(min(0.05, self.settings.lock_timeout_seconds - elapsed))
            report["lock_wait_seconds"] = time.monotonic() - wait_started
            if not acquired:
                status = "lock_unavailable" if lock_error else "lock_timeout"
                LOGGER.warning("Catalog cache %s; building privately", status)
                return finish(build_validated(), status)
            matrix = lookup()
            if matrix is not None:
                return finish(matrix, "hit_after_wait")
            matrix = build_validated()
            try:
                write_cache(path, key, names, matrix)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                LOGGER.warning("Catalog cache publication failed: %s", type(exc).__name__)
                return finish(matrix, "write_failed")
            return finish(matrix, "built")
        finally:
            # Closing the fd releases flock even on model errors or termination.
            os.close(lock_fd)
