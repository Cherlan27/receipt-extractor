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

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("12", 12.0),
            ("€1,99", 1.99),
            ("1,99 EUR", 1.99),
            ("1,99 eur", 1.99),
            ("$ 1.99", 1.99),
            ("-0,25", -0.25),
            ("1.234,56", 1234.56),
            ("12.345.678,90", 12345678.90),
            ("1,234.56", 1234.56),
            ("1.234,5 €", 1234.5),
        ],
    )
    def test_parses_supported_formats(self, raw, expected):
        item = ReceiptItem(product="X", price=raw, category="Obst")

        assert item.price == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw",
        [
            "2 x 1,99",  # quantity line, would silently become 21.99
            "2x1,99",
            "19,99 B 2",  # price plus tax class and quantity
            "1e3",
            "12 345",
            "1.234",  # three decimals / ambiguous thousands group
            "1,99-",
            "1,2,3",
            "",
            "   ",
            "nan",
            "inf",
        ],
    )
    def test_rejects_ambiguous_or_malformed_prices(self, raw):
        with pytest.raises(ValidationError):
            ReceiptItem(product="X", price=raw, category="Obst")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_numbers(self, value):
        with pytest.raises(ValidationError):
            ReceiptItem(product="X", price=value, category="Obst")

    def test_nan_from_model_json_is_rejected(self):
        with pytest.raises(ValidationError):
            ReceiptClassification.model_validate_json(
                '{"items": [{"product": "X", "price": NaN, "category": "Obst"}]}'
            )

    def test_rejects_boolean_price(self):
        with pytest.raises(ValidationError):
            ReceiptItem(product="X", price=True, category="Obst")

    def test_allows_negative_numeric_price_for_discounts_and_deposit_refunds(self):
        item = ReceiptItem(product="Pfand", price=-0.25, category="Anderes")

        assert item.price == -0.25


class TestReceiptItemCategory:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("obst", Category.FRUIT),
            ("OBST", Category.FRUIT),
            (" Obst ", Category.FRUIT),
            ("Obst.", Category.FRUIT),
            ("süßigkeiten", Category.SWEETS),
            ("SÜSSIGKEITEN", Category.SWEETS),
            ("gemüse", Category.VEGETABLES),
        ],
    )
    def test_matches_categories_leniently(self, raw, expected):
        item = ReceiptItem(product="X", price=1, category=raw)

        assert item.category == expected

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

    @pytest.mark.parametrize("product", ["   ", "\n\t"])
    def test_rejects_blank_product(self, product):
        with pytest.raises(ValidationError):
            ReceiptItem(product=product, price=1, category="Obst")

    def test_strips_surrounding_whitespace(self):
        item = ReceiptItem(product="  Apfel \n", price=1, category="Obst")

        assert item.product == "Apfel"


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
