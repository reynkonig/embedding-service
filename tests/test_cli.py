import os
import unittest
from unittest.mock import patch

from embedding_service.cli import build_parser, settings_from_args
from tests.helpers import TEST_DIMENSIONS, TEST_MODEL_ID, make_settings

REQUIRED_ARGUMENTS = (
    "--model",
    TEST_MODEL_ID,
    "--dimensions",
    str(TEST_DIMENSIONS),
)


class CommandLineConfigurationTest(unittest.TestCase):
    def parse(self, *arguments: str):
        return build_parser().parse_args(list(arguments))

    def test_device_alias_and_multi_device_flag_share_one_setting(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            single_device = settings_from_args(
                self.parse(*REQUIRED_ARGUMENTS, "--device", "cuda:2")
            )
            replicas = settings_from_args(
                self.parse(*REQUIRED_ARGUMENTS, "--devices", "cuda:0,cuda:3")
            )

        self.assertEqual(single_device.devices, ("cuda:2",))
        self.assertEqual(replicas.devices, ("cuda:0", "cuda:3"))

    def test_cli_values_override_environment_before_validation(self) -> None:
        environment = {
            "EMBEDDING_MODEL_ID": "environment/model",
            "EMBEDDING_DIMENSIONS": "128",
            "EMBEDDING_DEVICE": "cpu",
            "EMBEDDING_DEVICE_MAP": "auto",
            "EMBEDDING_DTYPE": "float32",
            "EMBEDDING_BATCH_SIZE": "2",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = settings_from_args(
                self.parse(
                    *REQUIRED_ARGUMENTS,
                    "--device",
                    "cuda:1",
                    "--dtype",
                    "float16",
                    "--batch-size",
                    "7",
                )
            )

        self.assertEqual(settings.model_id, TEST_MODEL_ID)
        self.assertEqual(settings.dimensions, TEST_DIMENSIONS)
        self.assertEqual(settings.devices, ("cuda:1",))
        self.assertIsNone(settings.device_map)
        self.assertEqual(settings.dtype, "float16")
        self.assertEqual(settings.batch_size, 7)

    def test_explicit_identity_uses_safe_generic_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = settings_from_args(self.parse(*REQUIRED_ARGUMENTS))

        self.assertIsNone(settings.model_revision)
        self.assertEqual(settings.backend, "mean")
        self.assertFalse(settings.trust_remote_code)
        self.assertEqual(settings.query_prefix, "")

    def test_model_and_dimensions_are_required_from_cli_or_environment(self) -> None:
        cases = (
            (("--model", TEST_MODEL_ID), "EMBEDDING_DIMENSIONS is required"),
            (("--dimensions", "2"), "EMBEDDING_MODEL_ID is required"),
        )
        for arguments, message in cases:
            with (
                self.subTest(arguments=arguments),
                patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(ValueError, message),
            ):
                settings_from_args(self.parse(*arguments))

        with patch.dict(
            os.environ,
            {"EMBEDDING_DIMENSIONS": "384"},
            clear=True,
        ):
            settings = settings_from_args(self.parse("--model", "example/encoder"))
        self.assertEqual(settings.dimensions, 384)

    def test_explicit_environment_policy_is_available_to_cli(self) -> None:
        environment = {
            "EMBEDDING_DIMENSIONS": "1024",
            "EMBEDDING_BACKEND": "cls",
            "EMBEDDING_TRUST_REMOTE_CODE": "false",
            "EMBEDDING_MODEL_REVISION": "release-7",
            "EMBEDDING_QUERY_PREFIX": "search: ",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = settings_from_args(self.parse("--model", "example/encoder"))

        self.assertEqual(settings.backend, "cls")
        self.assertEqual(settings.model_revision, "release-7")
        self.assertFalse(settings.trust_remote_code)
        self.assertEqual(settings.query_prefix, "search: ")

    def test_remote_code_and_revision_cli_values_validate_together(self) -> None:
        environment = {
            "EMBEDDING_MODEL_ID": TEST_MODEL_ID,
            "EMBEDDING_DIMENSIONS": str(TEST_DIMENSIONS),
            "EMBEDDING_TRUST_REMOTE_CODE": "true",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = settings_from_args(self.parse("--revision", "b" * 40))
        self.assertTrue(settings.trust_remote_code)
        self.assertEqual(settings.model_revision, "b" * 40)

    def test_cli_validation_still_runs_after_overrides(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "device_map"),
        ):
            settings_from_args(
                self.parse(
                    *REQUIRED_ARGUMENTS,
                    "--devices",
                    "cuda:0,cuda:1",
                    "--device-map",
                    "auto",
                )
            )

        self.assertEqual(make_settings().port, 8091)


if __name__ == "__main__":
    unittest.main()
