import math
import os
import unittest
from unittest.mock import patch

from embedding_service.config import DEFAULT_MAX_LENGTH, ServerSettings
from tests.helpers import TEST_DIMENSIONS, TEST_MODEL_ID, make_settings

REQUIRED_ENV = {
    "EMBEDDING_MODEL_ID": TEST_MODEL_ID,
    "EMBEDDING_DIMENSIONS": str(TEST_DIMENSIONS),
}


class ServerSettingsDefaultsTest(unittest.TestCase):
    def test_explicit_model_uses_neutral_runtime_defaults(self) -> None:
        settings = make_settings()

        self.assertEqual(settings.model_id, TEST_MODEL_ID)
        self.assertIsNone(settings.model_revision)
        self.assertEqual(settings.dimensions, TEST_DIMENSIONS)
        self.assertEqual(settings.backend, "mean")
        self.assertEqual(settings.query_prefix, "")
        self.assertEqual(settings.document_prefix, "")
        self.assertFalse(settings.trust_remote_code)
        self.assertTrue(settings.normalize_embeddings)
        self.assertEqual(settings.max_length, DEFAULT_MAX_LENGTH)
        self.assertEqual(settings.devices, ("auto",))
        self.assertEqual(settings.mode, "replicated")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8091)

    def test_device_map_selects_sharded_mode(self) -> None:
        self.assertEqual(make_settings(device_map="auto").mode, "sharded")

    def test_non_loopback_binding_requires_an_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "API_KEY"):
            make_settings(host="0.0.0.0")

        settings = make_settings(host="0.0.0.0", api_key="secret")
        self.assertEqual(settings.host, "0.0.0.0")


class ServerSettingsValidationTest(unittest.TestCase):
    def test_rejects_invalid_model_and_runtime_configuration(self) -> None:
        cases = {
            "empty model": {"model_id": " "},
            "dimension": {"dimensions": 0},
            "backend": {"backend": "unknown"},
            "dtype": {"dtype": "int8"},
            "attention": {"attn_implementation": "unknown"},
            "no devices": {"devices": ()},
            "duplicate devices": {"devices": ("cuda:0", "cuda:0")},
            "mixed auto": {"devices": ("auto", "cuda:0")},
            "sharded replicas": {
                "device_map": "auto",
                "devices": ("cuda:0",),
            },
            "batch size": {"batch_size": 0},
            "max length": {"max_length": 0},
            "max batch inputs": {"max_batch_inputs": 0},
            "max input characters": {"max_input_characters": 0},
            "max total characters": {"max_total_input_characters": 0},
            "pending requests": {"max_pending_requests": 0},
            "queue timeout": {"queue_timeout_seconds": 0},
            "inference timeout": {"inference_timeout_seconds": 0},
            "restart cooldown": {"worker_restart_cooldown_seconds": 0},
            "shutdown timeout": {"shutdown_timeout_seconds": 0},
            "normalization tolerance": {"normalization_tolerance": 0},
            "low port": {"port": 0},
            "high port": {"port": 65_536},
        }

        for name, values in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                make_settings(**values)

    def test_remote_code_requires_an_exact_commit_revision(self) -> None:
        for revision in (None, "main", "a" * 39, "g" * 40):
            with (
                self.subTest(revision=revision),
                self.assertRaisesRegex(ValueError, "40-character commit SHA"),
            ):
                make_settings(
                    trust_remote_code=True,
                    model_revision=revision,
                )

        settings = make_settings(
            trust_remote_code=True,
            model_revision="A" * 40,
        )
        self.assertEqual(settings.model_revision, "A" * 40)

    def test_rejects_non_finite_timeout_and_tolerance_values(self) -> None:
        fields = (
            "queue_timeout_seconds",
            "inference_timeout_seconds",
            "worker_restart_cooldown_seconds",
            "shutdown_timeout_seconds",
            "normalization_tolerance",
        )

        for field in fields:
            for value in (math.inf, -math.inf, math.nan):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    make_settings(**{field: value})


class ServerSettingsEnvironmentTest(unittest.TestCase):
    def test_model_and_dimensions_are_required(self) -> None:
        cases = (
            ({}, "EMBEDDING_MODEL_ID is required"),
            ({"EMBEDDING_MODEL_ID": TEST_MODEL_ID}, "EMBEDDING_DIMENSIONS is required"),
            ({"EMBEDDING_DIMENSIONS": "2"}, "EMBEDDING_MODEL_ID is required"),
        )

        for environment, message in cases:
            with (
                self.subTest(environment=environment),
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(ValueError, message),
            ):
                ServerSettings.from_env()

    def test_required_identity_gets_neutral_defaults(self) -> None:
        with patch.dict(os.environ, REQUIRED_ENV, clear=True):
            settings = ServerSettings.from_env()

        self.assertEqual(settings.model_id, TEST_MODEL_ID)
        self.assertEqual(settings.dimensions, TEST_DIMENSIONS)
        self.assertIsNone(settings.model_revision)
        self.assertFalse(settings.trust_remote_code)
        self.assertEqual(settings.backend, "mean")
        self.assertEqual(settings.query_prefix, "")

    def test_parses_complete_configuration(self) -> None:
        environment = {
            "EMBEDDING_MODEL_ID": " example/embedding-model ",
            "EMBEDDING_MODEL_REVISION": "release-1",
            "EMBEDDING_DIMENSIONS": "3",
            "EMBEDDING_BACKEND": "cls",
            "EMBEDDING_QUERY_PREFIX": "query:\\n",
            "EMBEDDING_DOCUMENT_PREFIX": "document:\\t",
            "EMBEDDING_TRUST_REMOTE_CODE": "false",
            "EMBEDDING_LOCAL_FILES_ONLY": "yes",
            "EMBEDDING_NORMALIZE": "off",
            "EMBEDDING_DEVICES": "cuda:0, cuda:2",
            "EMBEDDING_DTYPE": "float16",
            "EMBEDDING_BATCH_SIZE": "5",
            "EMBEDDING_MAX_LENGTH": "256",
            "EMBEDDING_ATTN_IMPLEMENTATION": "sdpa",
            "EMBEDDING_MAX_BATCH_INPUTS": "9",
            "EMBEDDING_MAX_INPUT_CHARACTERS": "101",
            "EMBEDDING_MAX_TOTAL_INPUT_CHARACTERS": "202",
            "EMBEDDING_MAX_PENDING_REQUESTS": "7",
            "EMBEDDING_QUEUE_TIMEOUT_SECONDS": "1.5",
            "EMBEDDING_INFERENCE_TIMEOUT_SECONDS": "8.5",
            "EMBEDDING_WORKER_RESTART_COOLDOWN_SECONDS": "2.5",
            "EMBEDDING_SHUTDOWN_TIMEOUT_SECONDS": "3.5",
            "EMBEDDING_NORMALIZATION_TOLERANCE": "0.005",
            "EMBEDDING_PRELOAD": "on",
            "EMBEDDING_API_KEY": " secret ",
            "EMBEDDING_HOST": "0.0.0.0",
            "EMBEDDING_PORT": "9000",
        }

        with patch.dict(os.environ, environment, clear=True):
            settings = ServerSettings.from_env()

        self.assertEqual(settings.model_id, TEST_MODEL_ID)
        self.assertEqual(settings.model_revision, "release-1")
        self.assertEqual(settings.dimensions, 3)
        self.assertEqual(settings.backend, "cls")
        self.assertEqual(settings.query_prefix, "query:\n")
        self.assertEqual(settings.document_prefix, "document:\t")
        self.assertFalse(settings.trust_remote_code)
        self.assertTrue(settings.local_files_only)
        self.assertFalse(settings.normalize_embeddings)
        self.assertEqual(settings.devices, ("cuda:0", "cuda:2"))
        self.assertEqual(settings.dtype, "float16")
        self.assertEqual(settings.batch_size, 5)
        self.assertEqual(settings.max_length, 256)
        self.assertEqual(settings.attn_implementation, "sdpa")
        self.assertEqual(settings.max_batch_inputs, 9)
        self.assertEqual(settings.max_input_characters, 101)
        self.assertEqual(settings.max_total_input_characters, 202)
        self.assertEqual(settings.max_pending_requests, 7)
        self.assertEqual(settings.queue_timeout_seconds, 1.5)
        self.assertEqual(settings.inference_timeout_seconds, 8.5)
        self.assertEqual(settings.worker_restart_cooldown_seconds, 2.5)
        self.assertEqual(settings.shutdown_timeout_seconds, 3.5)
        self.assertEqual(settings.normalization_tolerance, 0.005)
        self.assertTrue(settings.preload)
        self.assertEqual(settings.api_key, "secret")
        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 9000)

    def test_devices_variable_takes_precedence_over_device(self) -> None:
        environment = {
            **REQUIRED_ENV,
            "EMBEDDING_DEVICES": "cuda:1,cuda:3",
            "EMBEDDING_DEVICE": "cpu",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = ServerSettings.from_env()
        self.assertEqual(settings.devices, ("cuda:1", "cuda:3"))

    def test_singular_device_variable_is_supported(self) -> None:
        with patch.dict(
            os.environ,
            {**REQUIRED_ENV, "EMBEDDING_DEVICE": "cpu"},
            clear=True,
        ):
            settings = ServerSettings.from_env()
        self.assertEqual(settings.devices, ("cpu",))

    def test_overrides_are_applied_before_validation(self) -> None:
        environment = {
            **REQUIRED_ENV,
            "EMBEDDING_TRUST_REMOTE_CODE": "true",
            "EMBEDDING_DEVICES": "cuda:0,cuda:1",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = ServerSettings.from_env(
                model_revision="a" * 40,
                devices=("auto",),
                device_map="auto",
            )

        self.assertTrue(settings.trust_remote_code)
        self.assertEqual(settings.model_revision, "a" * 40)
        self.assertEqual(settings.mode, "sharded")

    def test_rejects_invalid_boolean_text(self) -> None:
        with (
            patch.dict(
                os.environ,
                {**REQUIRED_ENV, "EMBEDDING_PRELOAD": "sometimes"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "must be true or false"),
        ):
            ServerSettings.from_env()


if __name__ == "__main__":
    unittest.main()
