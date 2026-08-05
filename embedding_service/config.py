from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

DEFAULT_MAX_LENGTH = 512
SUPPORTED_DTYPES = {"auto", "float32", "float16", "bfloat16"}
SUPPORTED_BACKENDS = {"native", "mean", "cls", "last_token", "max"}
SUPPORTED_ATTENTION = {"auto", "eager", "sdpa", "flash_attention_2"}
COMMIT_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40}\Z")


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _boolean(name: str, default: bool) -> bool:
    value = _optional_text(os.getenv(name))
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _integer(name: str, default: int) -> int:
    value = _optional_text(os.getenv(name))
    return default if value is None else int(value)


def _float(name: str, default: float) -> float:
    value = _optional_text(os.getenv(name))
    return default if value is None else float(value)


def _prefix(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")


def _devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise ValueError("EMBEDDING_DEVICES must contain at least one device")
    return devices


@dataclass(frozen=True)
class ServerSettings:
    """Immutable service configuration.

    Every deployment explicitly selects one fixed model and output dimension.
    The request body cannot select an arbitrary Hugging Face repository.
    """

    model_id: str
    dimensions: int
    model_revision: str | None = None
    backend: str = "mean"
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False
    local_files_only: bool = False
    normalize_embeddings: bool = True
    devices: tuple[str, ...] = ("auto",)
    device_map: str | None = None
    dtype: str = "auto"
    batch_size: int = 8
    max_length: int = DEFAULT_MAX_LENGTH
    attn_implementation: str = "auto"
    max_batch_inputs: int = 64
    max_input_characters: int = 20_000
    max_total_input_characters: int = 200_000
    max_pending_requests: int = 32
    queue_timeout_seconds: float = 30.0
    inference_timeout_seconds: float = 900.0
    worker_restart_cooldown_seconds: float = 30.0
    shutdown_timeout_seconds: float = 30.0
    normalization_tolerance: float = 1e-2
    preload: bool = False
    api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8_091

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.dimensions < 1:
            raise ValueError("dimensions must be positive")
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend must be one of {', '.join(sorted(SUPPORTED_BACKENDS))}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype must be one of {', '.join(sorted(SUPPORTED_DTYPES))}"
            )
        if self.attn_implementation not in SUPPORTED_ATTENTION:
            raise ValueError(
                "attn_implementation must be one of "
                f"{', '.join(sorted(SUPPORTED_ATTENTION))}"
            )
        if not self.devices:
            raise ValueError("devices must contain at least one device")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("devices must not contain duplicates")
        if "auto" in self.devices and len(self.devices) != 1:
            raise ValueError("auto cannot be combined with explicit devices")
        if self.device_map is not None and self.devices != ("auto",):
            raise ValueError(
                "device_map shards one replica and cannot be combined with "
                "replicated devices"
            )
        if self.trust_remote_code and (
            self.model_revision is None
            or COMMIT_SHA_PATTERN.fullmatch(self.model_revision) is None
        ):
            raise ValueError(
                "trust_remote_code requires model_revision pinned to a full "
                "40-character commit SHA"
            )
        for name in (
            "batch_size",
            "max_length",
            "max_batch_inputs",
            "max_input_characters",
            "max_total_input_characters",
            "max_pending_requests",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "queue_timeout_seconds",
            "inference_timeout_seconds",
            "worker_restart_cooldown_seconds",
            "shutdown_timeout_seconds",
            "normalization_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.api_key:
            raise ValueError(
                "EMBEDDING_API_KEY is required when binding beyond loopback"
            )

    @property
    def mode(self) -> str:
        return "sharded" if self.device_map is not None else "replicated"

    @classmethod
    def from_env(cls, **overrides: object) -> ServerSettings:
        devices_value = (
            _optional_text(os.getenv("EMBEDDING_DEVICES"))
            or _optional_text(os.getenv("EMBEDDING_DEVICE"))
            or "auto"
        )
        dimensions_value = _optional_text(os.getenv("EMBEDDING_DIMENSIONS"))
        values: dict[str, object] = {
            "model_id": _optional_text(os.getenv("EMBEDDING_MODEL_ID")),
            "model_revision": _optional_text(os.getenv("EMBEDDING_MODEL_REVISION")),
            "dimensions": (
                int(dimensions_value) if dimensions_value is not None else None
            ),
            "backend": _optional_text(os.getenv("EMBEDDING_BACKEND")) or "mean",
            "query_prefix": _prefix("EMBEDDING_QUERY_PREFIX", ""),
            "document_prefix": _prefix("EMBEDDING_DOCUMENT_PREFIX", ""),
            "trust_remote_code": _boolean("EMBEDDING_TRUST_REMOTE_CODE", False),
            "local_files_only": _boolean("EMBEDDING_LOCAL_FILES_ONLY", False),
            "normalize_embeddings": _boolean("EMBEDDING_NORMALIZE", True),
            "devices": _devices(devices_value),
            "device_map": _optional_text(os.getenv("EMBEDDING_DEVICE_MAP")),
            "dtype": _optional_text(os.getenv("EMBEDDING_DTYPE")) or "auto",
            "batch_size": _integer("EMBEDDING_BATCH_SIZE", 8),
            "max_length": _integer("EMBEDDING_MAX_LENGTH", DEFAULT_MAX_LENGTH),
            "attn_implementation": (
                _optional_text(os.getenv("EMBEDDING_ATTN_IMPLEMENTATION")) or "auto"
            ),
            "max_batch_inputs": _integer("EMBEDDING_MAX_BATCH_INPUTS", 64),
            "max_input_characters": _integer("EMBEDDING_MAX_INPUT_CHARACTERS", 20_000),
            "max_total_input_characters": _integer(
                "EMBEDDING_MAX_TOTAL_INPUT_CHARACTERS", 200_000
            ),
            "max_pending_requests": _integer("EMBEDDING_MAX_PENDING_REQUESTS", 32),
            "queue_timeout_seconds": _float("EMBEDDING_QUEUE_TIMEOUT_SECONDS", 30.0),
            "inference_timeout_seconds": _float(
                "EMBEDDING_INFERENCE_TIMEOUT_SECONDS", 900.0
            ),
            "worker_restart_cooldown_seconds": _float(
                "EMBEDDING_WORKER_RESTART_COOLDOWN_SECONDS", 30.0
            ),
            "shutdown_timeout_seconds": _float(
                "EMBEDDING_SHUTDOWN_TIMEOUT_SECONDS", 30.0
            ),
            "normalization_tolerance": _float(
                "EMBEDDING_NORMALIZATION_TOLERANCE", 1e-2
            ),
            "preload": _boolean("EMBEDDING_PRELOAD", False),
            "api_key": _optional_text(os.getenv("EMBEDDING_API_KEY")),
            "host": _optional_text(os.getenv("EMBEDDING_HOST")) or "127.0.0.1",
            "port": _integer("EMBEDDING_PORT", 8_091),
        }
        values.update(overrides)

        model_id = values["model_id"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("EMBEDDING_MODEL_ID is required")
        if values["dimensions"] is None:
            raise ValueError("EMBEDDING_DIMENSIONS is required")
        return cls(**values)  # type: ignore[arg-type]
