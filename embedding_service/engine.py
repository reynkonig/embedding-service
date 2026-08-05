from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import ServerSettings

LOGGER = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """A model replica could not load or complete inference."""


class EmbeddingValidationError(RuntimeError):
    """A model returned an output that violates its configured contract."""


@dataclass(frozen=True)
class EngineResult:
    embeddings: Sequence[Sequence[float]]
    prompt_tokens: int


class TransformersEmbeddingEngine:
    """Lazy, serialized inference for one device or one sharded model."""

    def __init__(self, settings: ServerSettings, device: str) -> None:
        self.settings = settings
        self.requested_device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._input_device: Any = None
        self._resolved_device: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._last_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def snapshot(self) -> dict[str, object]:
        return {
            "device": self.requested_device,
            "resolved_device": self._resolved_device,
            "loaded": self.loaded,
            "state": "ready"
            if self.loaded
            else ("failed" if self._last_error else "cold"),
            "last_error": self._last_error,
        }

    def _resolve_runtime(self) -> tuple[Any | None, Any]:
        import torch

        if self.settings.device_map is not None:
            device = None
            cuda_device = torch.device("cuda:0") if torch.cuda.is_available() else None
        else:
            requested = self.requested_device
            if requested == "auto":
                if torch.cuda.is_available():
                    requested = "cuda:0"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    requested = "mps"
                else:
                    requested = "cpu"
            device = torch.device(requested)
            cuda_device = device if device.type == "cuda" else None
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise EmbeddingServiceError(
                        f"CUDA device {requested!r} was requested but CUDA is unavailable"
                    )
                index = device.index if device.index is not None else 0
                if index >= torch.cuda.device_count():
                    raise EmbeddingServiceError(
                        f"CUDA device {requested!r} does not exist; "
                        f"visible device count is {torch.cuda.device_count()}"
                    )

        dtype_by_name = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.settings.dtype == "auto":
            if cuda_device is None:
                dtype = torch.float32
            else:
                with torch.cuda.device(cuda_device):
                    supports_bfloat16 = torch.cuda.is_bf16_supported()
                dtype = torch.bfloat16 if supports_bfloat16 else torch.float16
        else:
            dtype = dtype_by_name[self.settings.dtype]

        return device, dtype

    @staticmethod
    def _first_model_device(model: Any) -> Any:
        try:
            embeddings = model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None and str(weight.device) != "meta":
                return weight.device
        except (AttributeError, NotImplementedError):
            pass
        device = getattr(model, "device", None)
        if device is None or str(device) == "meta":
            raise EmbeddingServiceError(
                "could not determine the input device for the sharded model"
            )
        return device

    def load(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModel, AutoTokenizer

                device, dtype = self._resolve_runtime()
                common_kwargs: dict[str, object] = {
                    "trust_remote_code": self.settings.trust_remote_code,
                    "local_files_only": self.settings.local_files_only,
                }
                if self.settings.model_revision is not None:
                    common_kwargs["revision"] = self.settings.model_revision

                tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_id,
                    **common_kwargs,
                )
                if (
                    getattr(tokenizer, "pad_token_id", None) is None
                    and getattr(tokenizer, "eos_token", None) is not None
                ):
                    # Decoder-only models commonly omit a padding token even
                    # though batched embedding inference requires one.
                    tokenizer.pad_token = tokenizer.eos_token
                model_kwargs = {
                    **common_kwargs,
                    "dtype": dtype,
                }
                if self.settings.attn_implementation != "auto":
                    model_kwargs["attn_implementation"] = (
                        self.settings.attn_implementation
                    )
                if self.settings.device_map is not None:
                    model_kwargs["device_map"] = self.settings.device_map

                model = AutoModel.from_pretrained(
                    self.settings.model_id,
                    **model_kwargs,
                )
                model.eval()
                if device is not None:
                    model.to(device)
                    input_device = device
                    resolved_device = str(device)
                else:
                    input_device = self._first_model_device(model)
                    resolved_device = f"device_map:{self.settings.device_map}"
            except EmbeddingServiceError as error:
                self._last_error = type(error).__name__
                raise
            except Exception as error:
                self._last_error = type(error).__name__
                LOGGER.exception(
                    "Failed to load embedding model %s on %s",
                    self.settings.model_id,
                    self.requested_device,
                )
                raise EmbeddingServiceError(
                    "failed to load the configured embedding model on "
                    f"{self.requested_device}"
                ) from error

            self._tokenizer = tokenizer
            self._model = model
            self._input_device = input_device
            self._resolved_device = resolved_device
            self._last_error = None

    @staticmethod
    def _native_tensor(output: Any) -> Any:
        import torch

        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for key in ("sentence_embedding", "embeddings", "embedding"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        for attribute in ("sentence_embedding", "embeddings", "embedding"):
            value = getattr(output, attribute, None)
            if torch.is_tensor(value):
                return value
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise EmbeddingValidationError(
            "native backend did not return a two-dimensional embedding tensor"
        )

    def _pool_tensor(self, output: Any, attention_mask: Any) -> Any:
        import torch

        if self.settings.backend == "native":
            tensor = self._native_tensor(output)
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, dict):
                hidden = output.get("last_hidden_state")
            if hidden is None and isinstance(output, (tuple, list)) and output:
                hidden = output[0]
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise EmbeddingValidationError(
                    f"{self.settings.backend} backend requires last_hidden_state"
                )

            if self.settings.backend == "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                tensor = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif self.settings.backend == "cls":
                tensor = hidden[:, 0]
            elif self.settings.backend == "last_token":
                reversed_mask = attention_mask.flip(dims=(1,))
                offsets = reversed_mask.argmax(dim=1)
                indexes = attention_mask.shape[1] - 1 - offsets
                rows = torch.arange(hidden.shape[0], device=hidden.device)
                tensor = hidden[rows, indexes]
            elif self.settings.backend == "max":
                mask = attention_mask.unsqueeze(-1).bool()
                tensor = (
                    hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min)
                    .max(dim=1)
                    .values
                )
            else:
                raise AssertionError("unsupported embedding backend")

        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise EmbeddingValidationError(
                "embedding backend did not return a two-dimensional tensor"
            )
        if self.settings.normalize_embeddings:
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)
        return tensor

    def embed(self, texts: Sequence[str]) -> EngineResult:
        self.load()

        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._input_device is not None

        vectors: list[list[float]] = []
        prompt_tokens = 0
        try:
            with self._inference_lock, torch.inference_mode():
                for start in range(0, len(texts), self.settings.batch_size):
                    batch_texts = list(texts[start : start + self.settings.batch_size])
                    encoded = self._tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.settings.max_length,
                        return_tensors="pt",
                    )
                    attention_mask = encoded["attention_mask"]
                    prompt_tokens += int(attention_mask.sum().item())
                    encoded = encoded.to(self._input_device)
                    if self.settings.backend == "native":
                        output = self._model(**encoded, return_embeddings=True)
                    else:
                        output = self._model(**encoded)
                    tensor = self._pool_tensor(output, encoded["attention_mask"])
                    vectors.extend(tensor.detach().float().cpu().tolist())
        except EmbeddingValidationError:
            raise
        except Exception as error:
            LOGGER.exception(
                "Embedding inference failed for %s on %s",
                self.settings.model_id,
                self.requested_device,
            )
            raise EmbeddingServiceError(
                f"embedding inference failed on {self.requested_device}"
            ) from error

        return EngineResult(embeddings=vectors, prompt_tokens=prompt_tokens)
