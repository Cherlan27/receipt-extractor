import pytest
from pydantic import ValidationError

from receipt_extractor.models.receipt import (
    Category,
    ClassifyTextRequest,
    ReceiptClassification,
    ReceiptItem,
)


class TestReceiptItemPrice:
    def test_accepts_numeric_price(self):
        item = ReceiptItem(product="Apfel", price=1.99, category="Obst")

        assert item.price == 1.99

    def test_accepts_dot_decimal_string(self):
        item = ReceiptItem(product="Apfel", price="1.99", category="Obst")

        assert item.price == 1.99

    def test_normalizes_comma_decimal_string(self):
        item = ReceiptItem(product="Apfel", price="1,99", category="Obst")

        assert item.price == 1.99

    def test_strips_currency_symbols_and_whitespace(self):
        item = ReceiptItem(product="Apfel", price=" 1,99 € ", category="Obst")

        assert item.price == 1.99

    def test_rejects_non_numeric_price(self):
        with pytest.raises(ValidationError):
            ReceiptItem(product="Apfel", price="gratis", category="Obst")


class TestReceiptItemCategory:
    @pytest.mark.parametrize("category", [c.value for c in Category])
    def test_accepts_every_allowed_category(self, category):
        item = ReceiptItem(product="X", price=1, category=category)

        assert item.category == category

    def test_unknown_category_falls_back_to_anderes(self):
        item = ReceiptItem(product="Joghurt", price=0.5, category="Milchprodukte")

        assert item.category == Category.OTHER
        assert item.category == "Anderes"

    def test_non_string_category_falls_back_to_anderes(self):
        item = ReceiptItem(product="Joghurt", price=0.5, category=None)

        assert item.category == Category.OTHER


class TestReceiptItemProduct:
    def test_rejects_empty_product(self):
        with pytest.raises(ValidationError):
            ReceiptItem(product="", price=1, category="Obst")


class TestReceiptClassification:
    def test_preserves_item_order(self):
        result = ReceiptClassification.model_validate(
            {
                "items": [
                    {"product": "Apfel", "price": "1.99", "category": "Obst"},
                    {"product": "Milch", "price": "1.29", "category": "Anderes"},
                    {
                        "product": "Gummibärchen",
                        "price": "0.99",
                        "category": "Süßigkeiten",
                    },
                ]
            }
        )

        assert [i.product for i in result.items] == ["Apfel", "Milch", "Gummibärchen"]

    def test_serializes_price_as_number_and_category_as_string(self):
        result = ReceiptClassification.model_validate(
            {"items": [{"product": "Apfel", "price": "1,99", "category": "Obst"}]}
        )

        assert result.model_dump(mode="json") == {
            "items": [{"product": "Apfel", "price": 1.99, "category": "Obst"}]
        }

    def test_allows_empty_items(self):
        assert ReceiptClassification(items=[]).items == []


class TestClassifyTextRequest:
    def test_accepts_text(self):
        assert ClassifyTextRequest(text="Apfel 1.99").text == "Apfel 1.99"

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_rejects_empty_or_whitespace_text(self, text):
        with pytest.raises(ValidationError):
            ClassifyTextRequest(text=text)

    def test_rejects_missing_text(self):
        with pytest.raises(ValidationError):
            ClassifyTextRequest()
