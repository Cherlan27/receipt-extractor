from unittest.mock import MagicMock, patch

import pytest
import torch

from receipt_extractor.llm.client import DEFAULT_MODEL_PATH, LlmClient
from receipt_extractor.models.llm_message import ChatMessage, PromptData


class _FakeEncoding(dict):
    """Minimal stand-in for a tokenizer's BatchEncoding: a dict with `.to()`."""

    def to(self, device):
        return self


@pytest.fixture
def patched_transformers():
    with (
        patch("receipt_extractor.llm.client.AutoTokenizer") as mock_tokenizer_cls,
        patch("receipt_extractor.llm.client.AutoModelForCausalLM") as mock_model_cls,
    ):
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock(name="tokenizer")
        mock_model = MagicMock(name="model")
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_tokenizer_cls, mock_model_cls


@pytest.fixture
def llm_client(patched_transformers, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PATH", "fake/model/path")
    return LlmClient()


def _prepare_generation(llm_client, new_token_ids=(4, 5), decoded="ok"):
    """Wire the mocked tokenizer/model so a 3-token prompt yields `new_token_ids`."""
    llm_client.tokenizer.apply_chat_template.return_value = "rendered-prompt"
    llm_client.tokenizer.return_value = _FakeEncoding(
        {"input_ids": torch.tensor([[1, 2, 3]])}
    )
    llm_client.model.generate.return_value = [torch.tensor([1, 2, 3, *new_token_ids])]
    llm_client.tokenizer.decode.return_value = decoded


class TestInit:
    def test_explicit_model_path_takes_precedence_over_env_var(
        self, patched_transformers, monkeypatch
    ):
        mock_tokenizer_cls, mock_model_cls = patched_transformers
        monkeypatch.setenv("LLM_MODEL_PATH", "from/env")

        client = LlmClient(model_path="explicit/path")

        assert client.model_path == "explicit/path"
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("explicit/path")
        assert mock_model_cls.from_pretrained.call_args.args == ("explicit/path",)

    def test_falls_back_to_env_var(self, patched_transformers, monkeypatch):
        mock_tokenizer_cls, mock_model_cls = patched_transformers
        monkeypatch.setenv("LLM_MODEL_PATH", "from/env")

        client = LlmClient()

        assert client.model_path == "from/env"
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("from/env")
        assert mock_model_cls.from_pretrained.call_args.args == ("from/env",)

    def test_falls_back_to_qwen_default_when_no_path_or_env_var(
        self, patched_transformers, monkeypatch
    ):
        mock_tokenizer_cls, _ = patched_transformers
        monkeypatch.delenv("LLM_MODEL_PATH", raising=False)

        client = LlmClient()

        assert DEFAULT_MODEL_PATH == "Qwen/Qwen3.5-9B"
        assert client.model_path == "Qwen/Qwen3.5-9B"
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("Qwen/Qwen3.5-9B")

    def test_loads_model_in_fp16_with_auto_device_map(self, patched_transformers):
        _, mock_model_cls = patched_transformers

        LlmClient(model_path="m")

        _, kwargs = mock_model_cls.from_pretrained.call_args
        assert kwargs["torch_dtype"] == torch.float16
        assert kwargs["device_map"] == "auto"
        mock_model_cls.from_pretrained.return_value.eval.assert_called_once()


class TestChat:
    def test_passes_messages_through_unchanged(self, llm_client):
        _prepare_generation(llm_client)
        messages = [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "Hi"},
        ]

        llm_client.chat(messages, max_new_tokens=5)

        template_args, template_kwargs = (
            llm_client.tokenizer.apply_chat_template.call_args
        )
        assert template_args[0] == messages
        assert template_kwargs["add_generation_prompt"] is True
        assert template_kwargs["enable_thinking"] is False

    def test_returns_only_newly_generated_tokens_decoded(self, llm_client):
        _prepare_generation(llm_client, new_token_ids=(4, 5), decoded="Bonjour")

        result = llm_client.chat([{"role": "user", "content": "Hi"}], max_new_tokens=5)

        assert result == "Bonjour"
        decoded_ids, decode_kwargs = llm_client.tokenizer.decode.call_args
        assert torch.equal(decoded_ids[0], torch.tensor([4, 5]))
        assert decode_kwargs["skip_special_tokens"] is True

    def test_uses_greedy_decoding_by_default(self, llm_client):
        _prepare_generation(llm_client)

        llm_client.chat([{"role": "user", "content": "Hi"}], max_new_tokens=5)

        _, generate_kwargs = llm_client.model.generate.call_args
        assert generate_kwargs["do_sample"] is False
        assert generate_kwargs["max_new_tokens"] == 5

    def test_sampling_can_be_enabled(self, llm_client):
        _prepare_generation(llm_client)

        llm_client.chat(
            [{"role": "user", "content": "Hi"}], max_new_tokens=5, do_sample=True
        )

        _, generate_kwargs = llm_client.model.generate.call_args
        assert generate_kwargs["do_sample"] is True

    def test_moves_inputs_to_model_device(self, llm_client):
        _prepare_generation(llm_client)
        encoding = MagicMock(wraps=_FakeEncoding({"input_ids": torch.tensor([[1]])}))
        encoding.to.return_value = _FakeEncoding({"input_ids": torch.tensor([[1]])})
        llm_client.tokenizer.return_value = encoding
        llm_client.model.generate.return_value = [torch.tensor([1, 2])]

        llm_client.chat([{"role": "user", "content": "Hi"}], max_new_tokens=5)

        encoding.to.assert_called_once_with(llm_client.model.device)


class TestGenerate:
    def test_returns_decoded_response(self, llm_client):
        _prepare_generation(llm_client, decoded="Bonjour")
        request = PromptData(
            messages=[ChatMessage(role="user", content="Hello")], max_new_tokens=42
        )

        assert llm_client.generate(request) == {"response": "Bonjour"}

    def test_does_not_inject_a_system_prompt(self, llm_client):
        _prepare_generation(llm_client)
        request = PromptData(messages=[ChatMessage(role="user", content="Hello")])

        llm_client.generate(request)

        template_args, _ = llm_client.tokenizer.apply_chat_template.call_args
        assert template_args[0] == [{"role": "user", "content": "Hello"}]

    def test_honors_client_supplied_system_message(self, llm_client):
        _prepare_generation(llm_client)
        request = PromptData(
            messages=[
                ChatMessage(role="system", content="Answer in French"),
                ChatMessage(role="user", content="Hello"),
            ]
        )

        llm_client.generate(request)

        template_args, _ = llm_client.tokenizer.apply_chat_template.call_args
        assert template_args[0][0] == {"role": "system", "content": "Answer in French"}
        assert template_args[0][1] == {"role": "user", "content": "Hello"}

    def test_uses_requested_max_new_tokens(self, llm_client):
        _prepare_generation(llm_client)
        request = PromptData(
            messages=[ChatMessage(role="user", content="Hi")], max_new_tokens=7
        )

        llm_client.generate(request)

        _, generate_kwargs = llm_client.model.generate.call_args
        assert generate_kwargs["max_new_tokens"] == 7

    def test_default_max_new_tokens_is_8192(self, llm_client):
        _prepare_generation(llm_client)
        request = PromptData(messages=[ChatMessage(role="user", content="Hi")])

        llm_client.generate(request)

        _, generate_kwargs = llm_client.model.generate.call_args
        assert generate_kwargs["max_new_tokens"] == 8192
