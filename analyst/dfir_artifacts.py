"""Safe, bounded artifact identity for offline DFIR fixtures."""
from __future__ import annotations

from pathlib import Path
import hashlib
import mimetypes
import uuid

from dfir_contracts import ArtifactRef


class ArtifactStore:
    def __init__(self, root, *, max_bytes=16_000_000):
        self.root = Path(root).resolve()
        self._artifacts = {}
        if not self.root.is_dir():
            raise ValueError("Artifact root must be an existing directory.")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be positive.")
        self.max_bytes = max_bytes

    def register(self, path) -> ArtifactRef:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ValueError("Artifact path escapes the configured root.") from None
        if not resolved.is_file():
            raise ValueError("Artifact must be a regular file.")
        size = resolved.stat().st_size
        if size > self.max_bytes:
            raise ValueError("Artifact exceeds the configured size limit.")
        digest = hashlib.sha256()
        with resolved.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        artifact = ArtifactRef(uuid.uuid4().hex, digest.hexdigest(), size, media_type, resolved.name, str(resolved))
        self._artifacts[artifact.artifact_id] = artifact
        return artifact

    def get(self, artifact_id):
        if not isinstance(artifact_id, str) or artifact_id not in self._artifacts:
            raise ValueError("Artifact is not registered.")
        return self._artifacts[artifact_id]

    def open(self, artifact_id):
        artifact = self.get(artifact_id)
        path = Path(artifact.storage_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise ValueError("Registered artifact escaped the configured root.") from None
        if not path.is_file():
            raise ValueError("Registered artifact is unavailable.")
        return path
