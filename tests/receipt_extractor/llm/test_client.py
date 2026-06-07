from unittest.mock import MagicMock, patch

import pytest
import torch

from receipt_extractor.llm.client import LlmClient
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
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_tokenizer_cls, mock_model_cls


@pytest.fixture
def llm_client(patched_transformers, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PATH", "fake/model/path")
    return LlmClient()


class TestInit:
    def test_explicit_model_path_takes_precedence_over_env_var(
        self, patched_transformers, monkeypatch
    ):
        mock_tokenizer_cls, _ = patched_transformers
        monkeypatch.setenv("LLM_MODEL_PATH", "from/env")

        client = LlmClient(model_path="explicit/path")

        assert client.model_path == "explicit/path"
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("explicit/path")

    def test_falls_back_to_env_var(self, patched_transformers, monkeypatch):
        mock_tokenizer_cls, _ = patched_transformers
        monkeypatch.setenv("LLM_MODEL_PATH", "from/env")

        client = LlmClient()

        assert client.model_path == "from/env"
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("from/env")

    def test_falls_back_to_default_when_no_path_or_env_var(
        self, patched_transformers, monkeypatch
    ):
        monkeypatch.delenv("LLM_MODEL_PATH", raising=False)

        client = LlmClient()

        assert client.model_path == "./openchat_model"


class TestSystemPrompt:
    def test_is_a_french_system_message(self, llm_client):
        assert llm_client.system_prompt == ChatMessage(
            role="system", content="Answer in French"
        )


class TestGenerate:
    def test_prepends_system_prompt_and_returns_decoded_response(self, llm_client):
        request = PromptData(
            messages=[ChatMessage(role="user", content="Hello")], max_new_tokens=42
        )

        prompt_inputs = _FakeEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
        llm_client.tokenizer.apply_chat_template.return_value = "rendered-prompt"
        llm_client.tokenizer.return_value = prompt_inputs
        llm_client.model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
        llm_client.tokenizer.decode.return_value = "Bonjour"

        result = llm_client.generate(request)

        assert result == {"response": "Bonjour"}

        template_args, template_kwargs = (
            llm_client.tokenizer.apply_chat_template.call_args
        )
        rendered_messages = template_args[0]
        assert rendered_messages[0] == {"role": "system", "content": "Answer in French"}
        assert rendered_messages[1] == {"role": "user", "content": "Hello"}
        assert template_kwargs["add_generation_prompt"] is True

        _, generate_kwargs = llm_client.model.generate.call_args
        assert generate_kwargs["max_new_tokens"] == 42

        decoded_ids, decode_kwargs = llm_client.tokenizer.decode.call_args
        assert torch.equal(decoded_ids[0], torch.tensor([4, 5]))
        assert decode_kwargs["skip_special_tokens"] is True

    def test_uses_requested_max_new_tokens(self, llm_client):
        request = PromptData(
            messages=[ChatMessage(role="user", content="Hi")], max_new_tokens=7
        )

        prompt_inputs = _FakeEncoding({"input_ids": torch.tensor([[1]])})
        llm_client.tokenizer.return_value = prompt_inputs
        llm_client.model.generate.return_value = [torch.tensor([1, 2])]
        llm_client.tokenizer.decode.return_value = "ok"

        llm_client.generate(request)

        _, generate_kwargs = llm_client.model.generate.call_args
        assert generate_kwargs["max_new_tokens"] == 7
