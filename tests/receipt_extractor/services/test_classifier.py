import json
import logging
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from receipt_extractor.models.receipt import Category, ReceiptClassification
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
            ("Apfel", Decimal("1.99"), Category.FRUIT),
            ("Milch", Decimal("1.29"), Category.OTHER),
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


def _output(items: str, total: str = "null") -> str:
    return f'{{"purchased_at": "2026-02-25", "store": "ALDI", "total": {total}, "items": [{items}]}}'


ITEM_APFEL = '{"product": "Apfel", "price": 1.99, "category": "Obst"}'
ITEM_MILCH = '{"product": "Milch", "price": 1.29, "category": "Anderes"}'
ITEM_A = '{"product": "A", "price": 0.1, "category": "Anderes"}'
ITEM_B = '{"product": "B", "price": 0.2, "category": "Anderes"}'


class TestMetadata:
    def test_parses_valid_metadata_and_quantity(self, classifier, llm_client):
        llm_client.chat.return_value = (
            '{"purchased_at": "25.02.26", "store": " ALDI ", "total": "21,89", '
            '"items": [{"product": "Käsebrötchen", "quantity": 2, '
            '"price": "2,78", "category": "Brot"}]}'
        )

        result = classifier.classify_text("ALDI")

        assert result.purchased_at == date(2026, 2, 25)
        assert result.store == "ALDI"
        assert result.total == Decimal("21.89")
        assert result.items[0].quantity == 2
        assert result.items[0].price == Decimal("2.78")

    @pytest.mark.parametrize(
        "raw_output",
        [
            '{"purchased_at": "25/02/2026", "items": []}',
            '{"total": "2 x 1,99", "items": []}',
            '{"store": 42, "items": []}',
            '{"items": [{"product": "A", "quantity": 0, "price": 1, "category": "Obst"}]}',
            '{"items": [{"product": "A", "quantity": 1.5, "price": 1, "category": "Obst"}]}',
        ],
    )
    def test_invalid_metadata_or_quantity_raises_classification_error(
        self, classifier, llm_client, raw_output
    ):
        llm_client.chat.return_value = raw_output

        with pytest.raises(ClassificationError) as exc_info:
            classifier.classify_text("ALDI")

        assert exc_info.value.raw_output == raw_output


class TestTotalMismatch:
    def _warnings(self, caplog):
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and r.name.endswith("classifier")
        ]

    def test_warns_with_both_amounts_when_sum_differs_from_total(
        self, classifier, llm_client, caplog
    ):
        llm_client.chat.return_value = _output(f"{ITEM_APFEL},{ITEM_MILCH}", "21.89")

        with caplog.at_level(logging.WARNING):
            result = classifier.classify_text("ALDI")

        (message,) = self._warnings(caplog)
        assert "3.28" in message
        assert "21.89" in message
        assert result.total == Decimal("21.89")
        assert [i.product for i in result.items] == ["Apfel", "Milch"]
        assert "warnings" not in result.model_dump()

    def test_no_warning_when_sum_matches_total(self, classifier, llm_client, caplog):
        llm_client.chat.return_value = _output(f"{ITEM_A},{ITEM_B}", "0.3")

        with caplog.at_level(logging.WARNING):
            classifier.classify_text("ALDI")

        assert self._warnings(caplog) == []

    def test_no_warning_when_total_is_null(self, classifier, llm_client, caplog):
        llm_client.chat.return_value = _output(f"{ITEM_APFEL},{ITEM_MILCH}")

        with caplog.at_level(logging.WARNING):
            classifier.classify_text("ALDI")

        assert self._warnings(caplog) == []


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
        assert '{"purchased_at": ' in system
        assert '"items": [' in system
        assert "{{" not in system
        assert "{format_instructions}" not in system
        assert "JSON schema" in system

    def _system_prompt(self, classifier, llm_client):
        llm_client.chat.return_value = VALID_JSON
        classifier.classify_text("Apfel 1.99")
        (messages,), _ = llm_client.chat.call_args
        return messages[0]["content"]

    @pytest.mark.parametrize(
        "needle",
        [
            '"purchased_at"',
            "YYYY-MM-DD",
            '"store"',
            '"total"',
            '"quantity"',
            "null",  # missing values
            "Gesamtpreis der Zeile",  # price is the line total
            "Steuerklasse",  # the digit after the euro sign
        ],
    )
    def test_system_prompt_asks_for_metadata_and_quantity(
        self, classifier, llm_client, needle
    ):
        assert needle in self._system_prompt(classifier, llm_client)

    def test_system_prompt_json_example_is_a_valid_classification(
        self, classifier, llm_client
    ):
        system = self._system_prompt(classifier, llm_client)

        example = system.split("Ausgabe:\n", 1)[1].split("\n\n", 1)[0]
        result = ReceiptClassification.model_validate(json.loads(example))
        assert result.store is not None
        assert result.purchased_at is not None
        assert result.total == sum(i.price for i in result.items)
        assert any(i.quantity > 1 for i in result.items)

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
