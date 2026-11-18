import logging
import re
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, PlainSerializer, field_validator

logger = logging.getLogger(__name__)


class Category(StrEnum):
    SWEETS = "Süßigkeiten"
    FRUIT = "Obst"
    VEGETABLES = "Gemüse"
    MEAT = "Fleisch"
    BREAD = "Brot"
    OTHER = "Anderes"


_CATEGORY_LOOKUP = {c.value.casefold(): c for c in Category}

# Accepted price formats, matched against the whole (trimmed) string:
# plain "1,99" / "1.99" / "12", German "1.234,56", English "1,234.56".
_PLAIN_PRICE = re.compile(r"-?\d+(?:[.,]\d{1,2})?")
_GERMAN_PRICE = re.compile(r"-?\d{1,3}(?:\.\d{3})+,\d{1,2}")
_ENGLISH_PRICE = re.compile(r"-?\d{1,3}(?:,\d{3})+\.\d{1,2}")


def parse_price(raw: str) -> Decimal:
    """
    Parse a price string as written on a receipt.

    Only whitespace and currency symbols around the number are ignored.
    Anything else (e.g. "2 x 1,99", "19,99 B 2", "1e3") is rejected instead of
    being squeezed into a wrong number.

    Raises:
        ValueError: If the string is not a recognized price format
    """
    text = re.sub(r"^[€$£\s]+", "", raw)
    text = re.sub(r"(?i)(?:[€$£\s]|EUR)+$", "", text)

    if _GERMAN_PRICE.fullmatch(text):
        return Decimal(text.replace(".", "").replace(",", "."))
    if _ENGLISH_PRICE.fullmatch(text):
        return Decimal(text.replace(",", ""))
    if _PLAIN_PRICE.fullmatch(text):
        return Decimal(text.replace(",", "."))
    raise ValueError(f"Unrecognized price format: {raw!r}")


def normalize_money(value):
    """
    Convert a model-reported amount to an exact Decimal.

    Numbers go through str() so that 1.99 stays 1.99 and does not become
    1.9899999...; strings use parse_price().

    Raises:
        ValueError: If the value is a boolean or not a recognized amount
    """
    if isinstance(value, bool):
        raise ValueError("amount must be a number")
    if isinstance(value, str):
        return parse_price(value)
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    return value


# Exact Decimal in Python, plain JSON number (not a string) in responses.
Money = Annotated[
    Decimal,
    Field(allow_inf_nan=False),
    PlainSerializer(float, return_type=float, when_used="json"),
]


# Accepted purchase dates, matched against the whole (trimmed) string, with an
# optional time: ISO "2026-02-25[T| ]19:33[:55]" and German "25.02.26[ 19:33]".
_TIME = r"(?P<h>\d{2}):(?P<min>\d{2})(?::(?P<s>\d{2}))?"
_ISO_DATE = re.compile(rf"(?P<y>\d{{4}})-(?P<m>\d{{2}})-(?P<d>\d{{2}})(?:[T ]{_TIME})?")
_GERMAN_DATE = re.compile(
    rf"(?P<d>\d{{2}})\.(?P<m>\d{{2}})\.(?P<y>\d{{4}}|\d{{2}})(?: {_TIME})?"
)


def parse_purchased_at(raw: str) -> date | datetime:
    """
    Parse a purchase date as reported by the model.

    Returns a date, or a naive datetime when the string carries a time.

    Raises:
        ValueError: If the string is not a recognized format or not a real date
    """
    text = raw.strip()
    match = _ISO_DATE.fullmatch(text) or _GERMAN_DATE.fullmatch(text)
    if match is None:
        raise ValueError(f"Unrecognized date format: {raw!r}")

    year = int(match["y"])
    if len(match["y"]) == 2:
        year += 2000
    day = date(year, int(match["m"]), int(match["d"]))
    if match["h"] is None:
        return day
    return datetime(
        day.year,
        day.month,
        day.day,
        int(match["h"]),
        int(match["min"]),
        int(match["s"] or 0),
    )


class ReceiptItem(BaseModel):
    product: str = Field(min_length=1)
    quantity: int = Field(default=1, ge=1)
    price: Money
    category: Category

    @field_validator("product", mode="before")
    @classmethod
    def strip_product(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("price", mode="before")
    @classmethod
    def normalize_price(cls, value):
        return normalize_money(value)

    @field_validator("quantity", mode="before")
    @classmethod
    def normalize_quantity(cls, value):
        if value is None:
            return 1
        if isinstance(value, bool):
            raise ValueError("quantity must be an integer")
        if isinstance(value, str):
            if not re.fullmatch(r"\d+", value.strip()):
                raise ValueError(f"quantity must be an integer: {value!r}")
            return int(value)
        if isinstance(value, float):
            raise ValueError("quantity must be an integer")
        return value

    @field_validator("category", mode="before")
    @classmethod
    def fallback_unknown_category(cls, value):
        if isinstance(value, str):
            key = value.strip().rstrip(".").strip().casefold()
            if key in _CATEGORY_LOOKUP:
                return _CATEGORY_LOOKUP[key]
        logger.warning("Unknown category %r, falling back to %s", value, Category.OTHER)
        return Category.OTHER


class ReceiptClassification(BaseModel):
    purchased_at: date | datetime | None = None
    store: str | None = None
    total: Money | None = None
    items: list[ReceiptItem]

    @field_validator("purchased_at", mode="before")
    @classmethod
    def parse_date(cls, value):
        if isinstance(value, str):
            return parse_purchased_at(value) if value.strip() else None
        if value is None or isinstance(value, date):
            return value
        raise ValueError("purchased_at must be a date string")

    @field_validator("store", mode="before")
    @classmethod
    def strip_store(cls, value):
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("store must be text")
        return value.strip() or None

    @field_validator("total", mode="before")
    @classmethod
    def normalize_total(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return normalize_money(value)


class ReceiptClassificationResponse(ReceiptClassification):
    raw_text: str


class ClassifyTextRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value
