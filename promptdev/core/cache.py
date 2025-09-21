import contextlib
import hashlib
import time
from pathlib import Path
from typing import Any

from pydantic_core import to_json

from promptdev.utils.file import read_json_file, write_json_file


class CacheManager:
    def __init__(self, enabled: bool = True, cache_dir: Path | None = None):
        self.cache = SimpleCache(enabled, cache_dir)


class SimpleCache:
    """Simple file-based cache for storing evaluation results."""

    def __init__(self, enabled: bool = True, cache_dir: Path | None = None):
        """Initialize the cache.

        Args:
            enabled: Whether the cache is enabled
            cache_dir: Directory to store cache files (defaults to ~/.promptdev/cache)
        """
        self.enabled = enabled

        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path.home() / ".promptdev" / "cache"
        else:
            self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file for storing all entries
        self.cache_file = self.cache_dir / "promptdev_cache.json"

    def generate_cache_key(
        self,
        model: str,
        prompt_content: str,
        variables: dict[str, Any],
        provider_config: dict[str, Any] | None = None,
    ) -> str:
        """Generate a cache key from evaluation parameters.

        Args:
            model: Model identifier
            prompt_content: The actual prompt content/template
            variables: Test case variables
            provider_config: Provider configuration (temperature, etc.)

        Returns:
            Cache key string
        """
        # Create a deterministic key from the inputs
        cache_data = {
            "model": model,
            "prompt_content": prompt_content,
            "variables": variables,
            "provider_config": provider_config or {},
        }

        # Sort keys for consistent hashing (pydantic_core sorts by default)
        cache_json = to_json(cache_data).decode("utf-8")

        # Generate SHA256 hash for the key
        return hashlib.sha256(cache_json.encode()).hexdigest()

    def _load_cache(self) -> dict[str, Any]:
        """Load cache data from file.

        Returns:
            Dictionary of cached data
        """
        if not self.enabled or not self.cache_file.exists():
            return {}

        try:
            cache_data = read_json_file(self.cache_file)

            # Check for TTL expiration if enabled
            current_time = time.time()
            valid_cache = {}

            for key, entry in cache_data.items():
                # Entry format: {"value": ..., "timestamp": ..., "ttl": ...}
                if isinstance(entry, dict) and "timestamp" in entry:
                    timestamp = entry["timestamp"]
                    ttl = entry.get("ttl")

                    # Check if entry has expired
                    if ttl is None or (current_time - timestamp) < ttl:
                        valid_cache[key] = entry
                    # else: entry has expired, don't include it
                else:
                    # Legacy format without timestamp, keep it
                    valid_cache[key] = {"value": entry, "timestamp": current_time}

            return valid_cache

        except (OSError, ValueError, KeyError):
            # If cache file is corrupted, start fresh
            return {}

    def _save_cache(self, cache_data: dict[str, Any]) -> None:
        """Save cache data to file.

        Args:
            cache_data: Dictionary of cache data to save
        """
        if not self.enabled:
            return

        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Write cache data with atomic operation
            temp_file = self.cache_file.with_suffix(".tmp")
            write_json_file(temp_file, cache_data)

            # Atomic rename
            temp_file.replace(self.cache_file)

        except OSError:
            pass

    def get(self, cache_key: str) -> Any | None:
        """Get a value from the cache.

        Args:
            cache_key: The cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None

        cache_data = self._load_cache()
        entry = cache_data.get(cache_key)

        if entry is None:
            return None

        # Extract value from entry
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        # Legacy format
        return entry

    def set(self, cache_key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in the cache.

        Args:
            cache_key: The cache key
            value: The value to cache
            ttl: Time to live in seconds (optional)
        """
        if not self.enabled:
            return

        # Load existing cache
        cache_data = self._load_cache()

        # Create new entry with timestamp
        entry = {"value": value, "timestamp": time.time()}

        if ttl is not None:
            entry["ttl"] = ttl

        cache_data[cache_key] = entry

        # Save back to file
        self._save_cache(cache_data)

    def clear(self) -> None:
        """Clear all cached values."""
        if not self.enabled:
            return

        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
        except OSError:
            pass

    def size(self) -> int:
        """Get the number of cached items."""
        if not self.enabled:
            return 0

        cache_data = self._load_cache()
        return len(cache_data)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled:
            return {
                "enabled": False,
                "size": 0,
                "cache_file": str(self.cache_file),
                "cache_file_exists": False,
                "keys": [],
            }

        cache_data = self._load_cache()
        file_size = 0
        if self.cache_file.exists():
            with contextlib.suppress(OSError):
                file_size = self.cache_file.stat().st_size

        return {
            "enabled": self.enabled,
            "size": len(cache_data),
            "cache_file": str(self.cache_file),
            "cache_file_exists": self.cache_file.exists(),
            "cache_file_size_bytes": file_size,
            "keys": list(cache_data.keys())[:10],  # Show first 10 keys for debugging
        }
