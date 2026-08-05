from __future__ import annotations

import asyncio
import hmac
import math
from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import ServerSettings
from .engine import EmbeddingServiceError, EmbeddingValidationError, EngineResult
from .pool import EmbeddingBackend, EmbeddingBusyError, EmbeddingProcessPool


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    input_type: Literal["document", "query"] = "document"
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = None
    user: str | None = None


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    model_revision: str | None
    usage: EmbeddingUsage


class ModelData(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelData]


def validate_inputs(value: str | list[str], settings: ServerSettings) -> list[str]:
    texts = [value] if isinstance(value, str) else list(value)
    if not texts:
        raise ValueError("input must contain at least one string")
    if len(texts) > settings.max_batch_inputs:
        raise ValueError(
            f"input must contain at most {settings.max_batch_inputs} strings"
        )
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("input strings must not be empty")
    if any(len(text) > settings.max_input_characters for text in texts):
        raise ValueError(
            "each input must contain at most "
            f"{settings.max_input_characters} characters"
        )
    if sum(len(text) for text in texts) > settings.max_total_input_characters:
        raise ValueError(
            "combined input must contain at most "
            f"{settings.max_total_input_characters} characters"
        )
    return texts


def prepare_inputs(
    value: str | list[str],
    input_type: str,
    settings: ServerSettings,
) -> list[str]:
    texts = validate_inputs(value, settings)
    prefix = (
        settings.query_prefix if input_type == "query" else settings.document_prefix
    )
    prepared = [f"{prefix}{text}" for text in texts]
    if any(len(text) > settings.max_input_characters for text in prepared):
        raise ValueError(
            "an input exceeds the character limit after applying its configured prefix"
        )
    if sum(len(text) for text in prepared) > settings.max_total_input_characters:
        raise ValueError(
            "combined input exceeds the character limit after applying configured prefixes"
        )
    return prepared


def validate_embeddings(
    embeddings: Sequence[Sequence[float]],
    *,
    expected_count: int,
    expected_dimensions: int,
    require_normalized: bool,
    normalization_tolerance: float,
) -> list[list[float]]:
    if len(embeddings) != expected_count:
        raise EmbeddingValidationError(
            f"model returned {len(embeddings)} vectors for {expected_count} inputs"
        )

    checked: list[list[float]] = []
    for index, raw_vector in enumerate(embeddings):
        if len(raw_vector) != expected_dimensions:
            raise EmbeddingValidationError(
                f"embedding {index} has dimension {len(raw_vector)}; "
                f"expected {expected_dimensions}"
            )
        vector = [float(value) for value in raw_vector]
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingValidationError(
                f"embedding {index} contains a non-finite value"
            )
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm <= 1e-12:
            raise EmbeddingValidationError(f"embedding {index} has zero norm")
        if require_normalized and not math.isclose(
            norm,
            1.0,
            rel_tol=0.0,
            abs_tol=normalization_tolerance,
        ):
            raise EmbeddingValidationError(
                f"embedding {index} is not L2-normalized (norm={norm:.6f})"
            )
        checked.append(vector)
    return checked


def _backend_snapshot(backend: EmbeddingBackend) -> dict[str, object]:
    snapshot = backend.snapshot()
    return snapshot if isinstance(snapshot, dict) else {}


def create_app(
    settings: ServerSettings | None = None,
    *,
    backend: EmbeddingBackend | None = None,
) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    backend = backend or EmbeddingProcessPool(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if settings.preload:
                await asyncio.to_thread(backend.load)
            yield
        finally:
            await asyncio.to_thread(backend.close)

    app = FastAPI(
        title="Embedding Server",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.embedding_backend = backend
    app.state.embedding_settings = settings

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key is None:
            return
        expected = f"Bearer {settings.api_key}"
        if authorization is None or not hmac.compare_digest(
            authorization,
            expected,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    def health() -> dict[str, object]:
        snapshot = _backend_snapshot(backend)
        return {
            "status": "ok",
            "model": settings.model_id,
            "revision": settings.model_revision,
            "dimensions": settings.dimensions,
            "backend": settings.backend,
            "normalized": settings.normalize_embeddings,
            "loaded": backend.loaded,
            **snapshot,
        }

    @app.get("/readyz")
    def readiness() -> JSONResponse:
        snapshot = _backend_snapshot(backend)
        ready_workers = int(snapshot.get("ready_workers", int(backend.loaded)))
        failed_workers = int(snapshot.get("failed_workers", 0))
        worker_count = int(snapshot.get("worker_count", 1))
        ready = ready_workers > 0
        payload = {
            "status": "ok" if ready else "not_ready",
            "ready": ready,
            "ready_workers": ready_workers,
            "failed_workers": failed_workers,
            "worker_count": worker_count,
        }
        return JSONResponse(
            payload,
            status_code=(
                status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
        )

    @app.post("/admin/warmup", dependencies=[Depends(authenticate)])
    def warmup() -> dict[str, object]:
        try:
            backend.load()
        except EmbeddingServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except EmbeddingValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(error),
            ) from error
        return {"status": "ok", **_backend_snapshot(backend)}

    @app.get(
        "/v1/models",
        response_model=ModelList,
        dependencies=[Depends(authenticate)],
    )
    def models() -> ModelList:
        owner = settings.model_id.split("/", 1)[0]
        return ModelList(data=[ModelData(id=settings.model_id, owned_by=owner)])

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        dependencies=[Depends(authenticate)],
    )
    def embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        if request.model not in (None, settings.model_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only model {settings.model_id!r} is available",
            )
        if request.dimensions not in (None, settings.dimensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(f"Only {settings.dimensions}-dimensional output is available"),
            )
        try:
            texts = prepare_inputs(request.input, request.input_type, settings)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(error),
            ) from error

        try:
            result: EngineResult = backend.embed(texts)
        except EmbeddingBusyError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(error),
                headers={"Retry-After": "1"},
            ) from error
        except EmbeddingServiceError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except EmbeddingValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(error),
            ) from error

        try:
            checked = validate_embeddings(
                result.embeddings,
                expected_count=len(texts),
                expected_dimensions=settings.dimensions,
                require_normalized=settings.normalize_embeddings,
                normalization_tolerance=settings.normalization_tolerance,
            )
        except EmbeddingValidationError as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(error),
            ) from error

        return EmbeddingResponse(
            data=[
                EmbeddingData(embedding=vector, index=index)
                for index, vector in enumerate(checked)
            ],
            model=settings.model_id,
            model_revision=settings.model_revision,
            usage=EmbeddingUsage(
                prompt_tokens=result.prompt_tokens,
                total_tokens=result.prompt_tokens,
            ),
        )

    return app
