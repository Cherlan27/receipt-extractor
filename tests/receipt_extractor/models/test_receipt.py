from datetime import date, datetime
from decimal import Decimal

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

        assert item.price == Decimal("1.99")

    def test_accepts_dot_decimal_string(self):
        item = ReceiptItem(product="Apfel", price="1.99", category="Obst")

        assert item.price == Decimal("1.99")

    def test_normalizes_comma_decimal_string(self):
        item = ReceiptItem(product="Apfel", price="1,99", category="Obst")

        assert item.price == Decimal("1.99")

    def test_strips_currency_symbols_and_whitespace(self):
        item = ReceiptItem(product="Apfel", price=" 1,99 € ", category="Obst")

        assert item.price == Decimal("1.99")

    @pytest.mark.parametrize("raw", [0.1, 0.2, "0.1", "0.2"])
    def test_prices_are_exact_decimals(self, raw):
        item = ReceiptItem(product="X", price=raw, category="Obst")

        assert isinstance(item.price, Decimal)
        assert item.price == Decimal(str(raw))

    def test_sum_of_decimal_prices_is_exact(self):
        a = ReceiptItem(product="X", price=0.1, category="Obst")
        b = ReceiptItem(product="Y", price="0.2", category="Obst")

        assert a.price + b.price == Decimal("0.3")

    def test_json_output_uses_numbers_not_strings(self):
        item = ReceiptItem(product="X", price="0.1", category="Obst")

        assert item.model_dump(mode="json")["price"] == 0.1
        assert '"price":0.1' in item.model_dump_json()

    def test_rejects_non_numeric_price(self):
        with pytest.raises(ValidationError):
            ReceiptItem(product="Apfel", price="gratis", category="Obst")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("12", Decimal("12")),
            ("€1,99", Decimal("1.99")),
            ("1,99 EUR", Decimal("1.99")),
            ("1,99 eur", Decimal("1.99")),
            ("$ 1.99", Decimal("1.99")),
            ("-0,25", Decimal("-0.25")),
            ("1.234,56", Decimal("1234.56")),
            ("12.345.678,90", Decimal("12345678.90")),
            ("1,234.56", Decimal("1234.56")),
            ("1.234,5 €", Decimal("1234.5")),
        ],
    )
    def test_parses_supported_formats(self, raw, expected):
        item = ReceiptItem(product="X", price=raw, category="Obst")

        assert item.price == expected

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

        assert item.price == Decimal("-0.25")


class TestReceiptItemQuantity:
    def test_defaults_to_one_when_missing(self):
        item = ReceiptItem(product="X", price=1, category="Obst")

        assert item.quantity == 1

    def test_null_quantity_defaults_to_one(self):
        item = ReceiptItem(product="X", quantity=None, price=1, category="Obst")

        assert item.quantity == 1

    @pytest.mark.parametrize("raw", [2, "2", " 2 "])
    def test_accepts_integers_and_digit_strings(self, raw):
        item = ReceiptItem(product="X", quantity=raw, price="2,78", category="Obst")

        assert item.quantity == 2
        assert item.price == Decimal("2.78")

    @pytest.mark.parametrize("raw", [0, -1, 1.5, "2x", "1.5", "0", "-1", "", True])
    def test_rejects_invalid_quantity(self, raw):
        with pytest.raises(ValidationError):
            ReceiptItem(product="X", quantity=raw, price=1, category="Obst")


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

        assert result.model_dump(mode="json")["items"] == [
            {"product": "Apfel", "quantity": 1, "price": 1.99, "category": "Obst"}
        ]
        assert isinstance(result.model_dump(mode="json")["items"][0]["price"], float)

    def test_allows_empty_items(self):
        assert ReceiptClassification(items=[]).items == []

    def test_serialized_items_contain_all_fields_in_order(self):
        result = ReceiptClassification.model_validate(
            {
                "items": [
                    {"product": "Apfel", "price": 1.99, "category": "Obst"},
                    {"product": "Milch", "price": 1.29, "category": "Anderes"},
                    {"product": "Gummi", "price": 0.99, "category": "Süßigkeiten"},
                ]
            }
        )

        items = result.model_dump(mode="json")["items"]
        assert [i["product"] for i in items] == ["Apfel", "Milch", "Gummi"]
        for item in items:
            assert list(item) == ["product", "quantity", "price", "category"]
            assert isinstance(item["price"], float)


class TestReceiptMetadata:
    def test_metadata_is_parsed(self):
        result = ReceiptClassification.model_validate(
            {
                "purchased_at": "2026-02-25",
                "store": " ALDI ",
                "total": "21,89 €",
                "items": [],
            }
        )

        assert result.purchased_at == date(2026, 2, 25)
        assert result.store == "ALDI"
        assert result.total == Decimal("21.89")
        assert result.model_dump(mode="json") == {
            "purchased_at": "2026-02-25",
            "store": "ALDI",
            "total": 21.89,
            "items": [],
        }

    @pytest.mark.parametrize(
        ("raw", "parsed", "serialized"),
        [
            ("2026-02-25", date(2026, 2, 25), "2026-02-25"),
            ("25.02.2026", date(2026, 2, 25), "2026-02-25"),
            ("25.02.26", date(2026, 2, 25), "2026-02-25"),
            (" 2026-02-25 ", date(2026, 2, 25), "2026-02-25"),
            (
                "2026-02-25T19:33:55",
                datetime(2026, 2, 25, 19, 33, 55),
                "2026-02-25T19:33:55",
            ),
            (
                "2026-02-25 19:33",
                datetime(2026, 2, 25, 19, 33),
                "2026-02-25T19:33:00",
            ),
            (
                "2026-02-25T19:33",
                datetime(2026, 2, 25, 19, 33),
                "2026-02-25T19:33:00",
            ),
            (
                "25.02.26 19:33",
                datetime(2026, 2, 25, 19, 33),
                "2026-02-25T19:33:00",
            ),
            (
                "25.02.2026 19:33:55",
                datetime(2026, 2, 25, 19, 33, 55),
                "2026-02-25T19:33:55",
            ),
        ],
    )
    def test_parses_supported_date_formats(self, raw, parsed, serialized):
        result = ReceiptClassification(purchased_at=raw, items=[])

        assert result.purchased_at == parsed
        assert type(result.purchased_at) is type(parsed)
        assert result.model_dump(mode="json")["purchased_at"] == serialized

    @pytest.mark.parametrize(
        "raw",
        [
            "25/02/2026",
            "gestern",
            "2026-02-30",
            "30.02.2026",
            "19:33",
            "2026-02-25 25:00",
            "2026-02-25T19:33:55Z",
            "25.2.26",
            20260225,
            1.5,
            True,
            ["2026-02-25"],
        ],
    )
    def test_rejects_unparseable_date(self, raw):
        with pytest.raises(ValidationError):
            ReceiptClassification(purchased_at=raw, items=[])

    def test_store_is_trimmed(self):
        result = ReceiptClassification(store="  ALDI SÜD\n", items=[])

        assert result.store == "ALDI SÜD"

    @pytest.mark.parametrize("raw", [123, 1.5, True, ["ALDI"], {"name": "ALDI"}])
    def test_rejects_non_text_store(self, raw):
        with pytest.raises(ValidationError):
            ReceiptClassification(store=raw, items=[])

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("21,89", Decimal("21.89")),
            ("1.234,56 EUR", Decimal("1234.56")),
            (21.89, Decimal("21.89")),
            (0.3, Decimal("0.3")),
        ],
    )
    def test_total_uses_price_rules(self, raw, expected):
        result = ReceiptClassification(total=raw, items=[])

        assert result.total == expected
        assert isinstance(result.model_dump(mode="json")["total"], float)

    @pytest.mark.parametrize(
        "raw", ["2 x 1,99", "NaN", "nan", "inf", True, "1e3", "gratis"]
    )
    def test_rejects_unparseable_total(self, raw):
        with pytest.raises(ValidationError):
            ReceiptClassification(total=raw, items=[])

    @pytest.mark.parametrize("raw", [float("nan"), float("inf")])
    def test_rejects_non_finite_total(self, raw):
        with pytest.raises(ValidationError):
            ReceiptClassification(total=raw, items=[])

    def test_nan_total_from_model_json_is_rejected(self):
        with pytest.raises(ValidationError):
            ReceiptClassification.model_validate_json('{"total": NaN, "items": []}')

    @pytest.mark.parametrize("missing", ["omitted", None, "", "   "])
    def test_missing_metadata_yields_null(self, missing):
        data = {"items": []}
        if missing != "omitted":
            data.update(purchased_at=missing, store=missing, total=missing)

        result = ReceiptClassification.model_validate(data)

        assert result.model_dump(mode="json") == {
            "purchased_at": None,
            "store": None,
            "total": None,
            "items": [],
        }

    def test_metadata_is_parsed_from_model_json(self):
        result = ReceiptClassification.model_validate_json(
            '{"purchased_at": "2026-02-25T19:33:55", "store": "ALDI", '
            '"total": 21.89, "items": []}'
        )

        assert result.purchased_at == datetime(2026, 2, 25, 19, 33, 55)
        assert result.total == Decimal("21.89")


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
