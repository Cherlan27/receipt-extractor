from unittest.mock import MagicMock

import pytest

from receipt_extractor.models.receipt import Category
from receipt_extractor.services.classifier import (
    CLASSIFY_MAX_NEW_TOKENS,
    ClassificationError,
    ReceiptClassifier,
)

VALID_JSON = (
    '{"items": ['
    '{"product": "Apfel", "price": 1.99, "category": "Obst"},'
    '{"product": "Milch", "price": "1,29", "category": "Anderes"}'
    "]}"
)


@pytest.fixture
def llm_client():
    return MagicMock(name="LlmClient")


@pytest.fixture
def classifier(llm_client):
    return ReceiptClassifier(llm_client)


class TestClassifyText:
    def test_parses_valid_json_in_receipt_order(self, classifier, llm_client):
        llm_client.chat.return_value = VALID_JSON

        result = classifier.classify_text("Apfel 1.99\nMilch 1,29")

        assert [(i.product, i.price, i.category) for i in result.items] == [
            ("Apfel", 1.99, Category.FRUIT),
            ("Milch", 1.29, Category.OTHER),
        ]

    def test_accepts_json_wrapped_in_markdown_fence(self, classifier, llm_client):
        llm_client.chat.return_value = f"```json\n{VALID_JSON}\n```"

        result = classifier.classify_text("Apfel 1.99")

        assert len(result.items) == 2

    def test_empty_items_is_a_valid_result(self, classifier, llm_client):
        llm_client.chat.return_value = '{"items": []}'

        assert classifier.classify_text("nothing here").items == []

    def test_unknown_category_falls_back_to_anderes(self, classifier, llm_client):
        llm_client.chat.return_value = (
            '{"items": [{"product": "Joghurt", "price": 0.5, "category": "Milch"}]}'
        )

        result = classifier.classify_text("Joghurt 0.50")

        assert result.items[0].category == Category.OTHER

    @pytest.mark.parametrize(
        "raw_output",
        [
            "Hier ist deine Liste: Apfel, Milch",
            "",
            '{"items": [{"product": "Apfel"}]}',
            '[{"product": "Apfel", "price": 1.99, "category": "Obst"}]',
        ],
    )
    def test_invalid_output_raises_classification_error(
        self, classifier, llm_client, raw_output
    ):
        llm_client.chat.return_value = raw_output

        with pytest.raises(ClassificationError) as exc_info:
            classifier.classify_text("Apfel 1.99")

        assert exc_info.value.raw_output == raw_output


class TestPrompting:
    def test_sends_system_prompt_and_receipt_text_as_chat_messages(
        self, classifier, llm_client
    ):
        llm_client.chat.return_value = VALID_JSON

        classifier.classify_text("Apfel 1.99")

        (messages,), _ = llm_client.chat.call_args
        assert [m["role"] for m in messages] == ["system", "user"]
        assert (
            "Süßigkeiten, Obst, Gemüse, Fleisch, Brot, Anderes"
            in messages[0]["content"]
        )
        assert messages[1] == {"role": "user", "content": "Apfel 1.99"}

    def test_system_prompt_has_unescaped_json_example_and_format_instructions(
        self, classifier, llm_client
    ):
        llm_client.chat.return_value = VALID_JSON

        classifier.classify_text("Apfel 1.99")

        (messages,), _ = llm_client.chat.call_args
        system = messages[0]["content"]
        assert '{"items": [' in system
        assert "{{" not in system
        assert "{format_instructions}" not in system
        assert "JSON schema" in system

    def test_braces_in_receipt_text_are_not_treated_as_placeholders(
        self, classifier, llm_client
    ):
        llm_client.chat.return_value = VALID_JSON

        classifier.classify_text("Apfel {1.99} {receipt_text}")

        (messages,), _ = llm_client.chat.call_args
        assert messages[1]["content"] == "Apfel {1.99} {receipt_text}"

    def test_limits_generation_to_classify_max_new_tokens(self, classifier, llm_client):
        llm_client.chat.return_value = VALID_JSON

        classifier.classify_text("Apfel 1.99")

        _, kwargs = llm_client.chat.call_args
        assert kwargs["max_new_tokens"] == CLASSIFY_MAX_NEW_TOKENS == 4096
