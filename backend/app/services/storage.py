"""File storage service — local filesystem implementation.

Swappable to MinIO/S3 by replacing this module. All other code
uses the StorageService interface and never touches the filesystem directly.
"""

import os

from app.config import settings


class StorageService:
    """Local filesystem storage. Files stored under base_path/<key>."""

    def __init__(self, base_path: str | None = None):
        self.base_path = base_path or settings.storage_path

    def _full_path(self, key: str) -> str:
        return os.path.join(self.base_path, key.replace("/", os.sep))

    def save(self, key: str, content: bytes) -> str:
        path = self._full_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def read(self, key: str) -> bytes:
        path = self._full_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {key}")
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        path = self._full_path(key)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._full_path(key))


# Singleton instance
storage_service = StorageService()
