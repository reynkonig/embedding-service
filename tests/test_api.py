import math
import unittest

from fastapi.testclient import TestClient

from embedding_service.api import create_app
from embedding_service.engine import (
    EmbeddingServiceError,
    EmbeddingValidationError,
    EngineResult,
)
from embedding_service.pool import EmbeddingBusyError
from tests.helpers import (
    TEST_DIMENSIONS,
    TEST_MODEL_ID,
    TEST_MODEL_REVISION,
    make_settings,
)


def unit_vector(dimensions: int) -> list[float]:
    return [1.0, *([0.0] * (dimensions - 1))]


class FakeBackend:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.is_loaded = False
        self.load_calls = 0
        self.close_calls = 0
        self.embed_calls: list[list[str]] = []
        self.load_error: Exception | None = None
        self.embed_error: Exception | None = None
        self.result: EngineResult | None = None
        self.snapshot_values: dict[str, object] = {
            "mode": "fake",
            "worker_count": 1,
            "ready_workers": 0,
            "failed_workers": 0,
            "active_requests": 0,
            "available_workers": 1,
            "completed_requests": 0,
            "failed_requests": 0,
        }

    @property
    def loaded(self) -> bool:
        return self.is_loaded

    def load(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error
        self.is_loaded = True
        self.snapshot_values["ready_workers"] = 1

    def embed(self, texts: list[str]) -> EngineResult:
        self.embed_calls.append(list(texts))
        if self.embed_error is not None:
            raise self.embed_error
        self.is_loaded = True
        self.snapshot_values["ready_workers"] = 1
        if self.result is not None:
            return self.result
        return EngineResult(
            [unit_vector(self.dimensions) for _ in texts],
            prompt_tokens=7,
        )

    def snapshot(self) -> dict[str, object]:
        return dict(self.snapshot_values)

    def close(self) -> None:
        self.close_calls += 1


class EmbeddingApiContractTest(unittest.TestCase):
    def test_configured_model_uses_the_openai_response_contract(self) -> None:
        settings = make_settings(model_revision=TEST_MODEL_REVISION)
        backend = FakeBackend(TEST_DIMENSIONS)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": TEST_MODEL_ID, "input": "sample text"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(backend.embed_calls, [["sample text"]])
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], TEST_MODEL_ID)
        self.assertEqual(body["model_revision"], TEST_MODEL_REVISION)
        self.assertEqual(body["usage"], {"prompt_tokens": 7, "total_tokens": 7})
        self.assertEqual(body["data"][0]["object"], "embedding")
        self.assertEqual(body["data"][0]["index"], 0)
        self.assertEqual(len(body["data"][0]["embedding"]), TEST_DIMENSIONS)
        self.assertEqual(backend.close_calls, 1)

    def test_model_may_be_omitted(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post("/v1/embeddings", json={"input": "sample"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], TEST_MODEL_ID)

    def test_query_and_document_prefixes_are_applied_exactly_once(self) -> None:
        settings = make_settings(
            query_prefix="query: ",
            document_prefix="document: ",
        )
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            query = client.post(
                "/v1/embeddings",
                json={"input": ["alpha", "beta"], "input_type": "query"},
            )
            document = client.post(
                "/v1/embeddings",
                json={"input": "gamma", "input_type": "document"},
            )

        self.assertEqual(query.status_code, 200)
        self.assertEqual(document.status_code, 200)
        self.assertEqual(
            backend.embed_calls,
            [["query: alpha", "query: beta"], ["document: gamma"]],
        )
        self.assertEqual([row["index"] for row in query.json()["data"]], [0, 1])

    def test_query_input_is_unchanged_without_an_explicit_prefix(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post(
                "/v1/embeddings",
                json={"input": "complete client query", "input_type": "query"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(backend.embed_calls, [["complete client query"]])

    def test_rejects_wrong_model_and_dimensions_before_inference(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            wrong_model = client.post(
                "/v1/embeddings",
                json={"model": "other/model", "input": "sample"},
            )
            wrong_dimensions = client.post(
                "/v1/embeddings",
                json={"input": "sample", "dimensions": 3},
            )

        self.assertEqual(wrong_model.status_code, 400)
        self.assertEqual(wrong_dimensions.status_code, 400)
        self.assertEqual(backend.embed_calls, [])


class EmbeddingApiValidationTest(unittest.TestCase):
    def test_rejects_all_configured_input_limits_before_inference(self) -> None:
        settings = make_settings(
            max_batch_inputs=2,
            max_input_characters=4,
            max_total_input_characters=6,
        )
        cases = [
            {"input": []},
            {"input": "   "},
            {"input": ["a", "b", "c"]},
            {"input": "abcde"},
            {"input": ["abcd", "abcd"]},
            {"input": 123},
            {"input": "ok", "encoding_format": "base64"},
        ]

        for payload in cases:
            with self.subTest(payload=payload):
                backend = FakeBackend(2)
                with TestClient(create_app(settings, backend=backend)) as client:
                    response = client.post("/v1/embeddings", json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(backend.embed_calls, [])

    def test_rechecks_limits_after_adding_the_query_prefix(self) -> None:
        settings = make_settings(
            query_prefix="abc",
            max_input_characters=5,
            max_total_input_characters=20,
        )
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post(
                "/v1/embeddings",
                json={"input": "xyz", "input_type": "query"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("after applying", response.json()["detail"])
        self.assertEqual(backend.embed_calls, [])

    def test_maps_busy_and_service_failures_to_retryable_http_statuses(self) -> None:
        settings = make_settings()

        busy_backend = FakeBackend(2)
        busy_backend.embed_error = EmbeddingBusyError("queue full")
        with TestClient(create_app(settings, backend=busy_backend)) as client:
            busy = client.post("/v1/embeddings", json={"input": "sample"})

        failed_backend = FakeBackend(2)
        failed_backend.embed_error = EmbeddingServiceError("GPU unavailable")
        with TestClient(create_app(settings, backend=failed_backend)) as client:
            failed = client.post("/v1/embeddings", json={"input": "sample"})

        self.assertEqual(busy.status_code, 429)
        self.assertEqual(busy.headers["retry-after"], "1")
        self.assertEqual(busy.json()["detail"], "queue full")
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(failed.json()["detail"], "GPU unavailable")

    def test_maps_backend_validation_failures_to_internal_server_error(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)
        backend.embed_error = EmbeddingValidationError("bad backend tensor")

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post("/v1/embeddings", json={"input": "sample"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "bad backend tensor")

    def test_rejects_invalid_model_outputs(self) -> None:
        settings = make_settings()
        cases = {
            "cardinality": EngineResult([], 1),
            "dimensions": EngineResult([[1.0]], 1),
            "non-finite": EngineResult([[math.nan, 0.0]], 1),
            "zero norm": EngineResult([[0.0, 0.0]], 1),
            "not normalized": EngineResult([[2.0, 0.0]], 1),
        }

        for name, result in cases.items():
            with self.subTest(name=name):
                backend = FakeBackend(2)
                backend.result = result
                with TestClient(create_app(settings, backend=backend)) as client:
                    response = client.post(
                        "/v1/embeddings",
                        json={"input": "sample"},
                    )
                self.assertEqual(response.status_code, 500)

    def test_accepts_finite_non_unit_vectors_when_normalization_is_disabled(
        self,
    ) -> None:
        settings = make_settings(normalize_embeddings=False)
        backend = FakeBackend(2)
        backend.result = EngineResult([[2.0, 0.0]], 1)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post("/v1/embeddings", json={"input": "sample"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["embedding"], [2.0, 0.0])


class EmbeddingApiOperationsTest(unittest.TestCase):
    def test_health_is_lightweight_and_preserves_required_identity_fields(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model"], settings.model_id)
        self.assertEqual(body["revision"], settings.model_revision)
        self.assertEqual(body["dimensions"], 2)
        self.assertEqual(body["backend"], "mean")
        self.assertTrue(body["normalized"])
        self.assertFalse(body["loaded"])
        self.assertEqual(body["mode"], "fake")
        self.assertEqual(backend.load_calls, 0)
        self.assertEqual(backend.embed_calls, [])

    def test_readiness_tracks_worker_availability_without_loading(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            cold = client.get("/readyz")
            backend.snapshot_values.update(
                {"ready_workers": 1, "failed_workers": 1, "worker_count": 2}
            )
            degraded = client.get("/readyz")

        self.assertEqual(cold.status_code, 503)
        self.assertEqual(cold.json()["status"], "not_ready")
        self.assertEqual(degraded.status_code, 200)
        self.assertTrue(degraded.json()["ready"])
        self.assertEqual(degraded.json()["failed_workers"], 1)
        self.assertEqual(backend.load_calls, 0)

    def test_models_endpoint_reports_only_the_configured_model(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.model_id,
                        "object": "model",
                        "owned_by": "example",
                    }
                ],
            },
        )

    def test_bearer_auth_protects_inference_models_and_warmup(self) -> None:
        settings = make_settings(api_key="secret")
        backend = FakeBackend(2)

        with TestClient(create_app(settings, backend=backend)) as client:
            health = client.get("/healthz")
            missing = client.post("/v1/embeddings", json={"input": "sample"})
            wrong = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong"},
            )
            valid = client.post(
                "/admin/warmup",
                headers={"Authorization": "Bearer secret"},
            )

        self.assertEqual(health.status_code, 200)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(backend.load_calls, 1)

    def test_warmup_failure_is_reported_as_unavailable(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)
        backend.load_error = EmbeddingServiceError("cannot load")

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post("/admin/warmup")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "cannot load")

    def test_warmup_validation_failure_is_reported_as_internal_error(self) -> None:
        settings = make_settings()
        backend = FakeBackend(2)
        backend.load_error = EmbeddingValidationError("invalid warmup output")

        with TestClient(create_app(settings, backend=backend)) as client:
            response = client.post("/admin/warmup")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "invalid warmup output")

    def test_preload_and_shutdown_run_through_the_application_lifespan(self) -> None:
        settings = make_settings(preload=True)
        backend = FakeBackend(2)

        self.assertEqual(backend.load_calls, 0)
        with TestClient(create_app(settings, backend=backend)) as client:
            self.assertEqual(backend.load_calls, 1)
            self.assertEqual(client.get("/readyz").status_code, 200)
            self.assertEqual(backend.close_calls, 0)

        self.assertEqual(backend.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
