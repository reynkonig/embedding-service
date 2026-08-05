import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from embedding_service.engine import TransformersEmbeddingEngine
from tests.helpers import make_settings


def engine_for(backend: str, *, normalize: bool = False):
    settings = make_settings(
        backend=backend,
        normalize_embeddings=normalize,
        trust_remote_code=False,
    )
    return TransformersEmbeddingEngine(settings, "cpu")


class PoolingBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
                [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            ]
        )
        self.mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        self.output = SimpleNamespace(last_hidden_state=self.hidden)

    def test_mean_pooling_ignores_padding(self) -> None:
        result = engine_for("mean")._pool_tensor(self.output, self.mask)
        torch.testing.assert_close(
            result,
            torch.tensor([[2.0, 3.0], [7.0, 8.0]]),
        )

    def test_cls_pooling_uses_the_first_token(self) -> None:
        result = engine_for("cls")._pool_tensor(self.output, self.mask)
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))

    def test_last_token_pooling_uses_the_last_unmasked_token(self) -> None:
        result = engine_for("last_token")._pool_tensor(self.output, self.mask)
        torch.testing.assert_close(result, torch.tensor([[3.0, 4.0], [9.0, 10.0]]))

    def test_max_pooling_ignores_padding(self) -> None:
        result = engine_for("max")._pool_tensor(self.output, self.mask)
        torch.testing.assert_close(result, torch.tensor([[3.0, 4.0], [9.0, 10.0]]))

    def test_native_backend_accepts_a_direct_embedding_tensor(self) -> None:
        output = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
        result = engine_for("native", normalize=True)._pool_tensor(
            output,
            self.mask,
        )
        torch.testing.assert_close(
            result,
            torch.tensor([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]]),
        )


class EncodedBatch(dict):
    def to(self, device):
        return EncodedBatch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, texts, **kwargs):
        values = list(texts)
        self.calls.append((values, dict(kwargs)))
        return EncodedBatch(
            {
                "input_ids": torch.ones((len(values), 2), dtype=torch.long),
                "attention_mask": torch.ones(
                    (len(values), 2),
                    dtype=torch.long,
                ),
            }
        )


class FakeModel:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.to_calls: list[object] = []
        self.forward_calls: list[dict[str, object]] = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self

    def __call__(self, **kwargs):
        self.forward_calls.append(dict(kwargs))
        rows = kwargs["input_ids"].shape[0]
        return torch.tensor([[1.0, 0.0]] * rows)


class LazyModelIntegrationTest(unittest.TestCase):
    def test_loads_once_and_uses_the_native_model_contract(self) -> None:
        settings = make_settings(
            backend="native",
            batch_size=2,
            max_length=17,
            dtype="float32",
            normalize_embeddings=False,
        )
        tokenizer = FakeTokenizer()
        model = FakeModel()

        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as tokenizer_loader,
            patch(
                "transformers.AutoModel.from_pretrained",
                return_value=model,
            ) as model_loader,
        ):
            engine = TransformersEmbeddingEngine(settings, "cpu")
            first = engine.embed(["one", "two", "three"])
            second = engine.embed(["four"])

        self.assertEqual(first.embeddings, [[1.0, 0.0]] * 3)
        self.assertEqual(first.prompt_tokens, 6)
        self.assertEqual(second.prompt_tokens, 2)
        self.assertEqual(tokenizer_loader.call_count, 1)
        self.assertEqual(model_loader.call_count, 1)
        self.assertEqual(model_loader.call_args.kwargs["dtype"], torch.float32)
        self.assertNotIn("torch_dtype", model_loader.call_args.kwargs)
        tokenizer_loader.assert_called_once_with(
            settings.model_id,
            trust_remote_code=False,
            local_files_only=False,
        )
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual([str(device) for device in model.to_calls], ["cpu"])
        self.assertEqual(
            [call[0] for call in tokenizer.calls],
            [["one", "two"], ["three"], ["four"]],
        )
        self.assertTrue(all(call[1]["max_length"] == 17 for call in tokenizer.calls))
        self.assertTrue(
            all(call["return_embeddings"] is True for call in model.forward_calls)
        )


if __name__ == "__main__":
    unittest.main()
