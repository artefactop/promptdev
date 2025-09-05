"""Simple file-based cache for PromptDev evaluations."""

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Any


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

        # Sort keys for consistent hashing
        cache_json = json.dumps(cache_data, sort_keys=True, ensure_ascii=True)

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
            with open(self.cache_file, encoding="utf-8") as f:
                cache_data = json.load(f)

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

        except (OSError, json.JSONDecodeError, KeyError) as e:
            # If cache file is corrupted, start fresh
            print(f"Warning: Cache file corrupted, starting fresh: {e}")
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
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(self.cache_file)

        except OSError as e:
            print(f"Warning: Could not save cache: {e}")

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
        except OSError as e:
            print(f"Warning: Could not clear cache: {e}")

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


# Global cache instance
_cache_instance: SimpleCache | None = None


def get_cache() -> SimpleCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SimpleCache()
    return _cache_instance


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable the global cache."""
    cache = get_cache()
    cache.enabled = enabled


def clear_cache() -> None:
    """Clear the global cache."""
    cache = get_cache()
    cache.clear()
