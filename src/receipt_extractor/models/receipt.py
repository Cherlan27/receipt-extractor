import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

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


def parse_price(raw: str) -> float:
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
        return float(text.replace(".", "").replace(",", "."))
    if _ENGLISH_PRICE.fullmatch(text):
        return float(text.replace(",", ""))
    if _PLAIN_PRICE.fullmatch(text):
        return float(text.replace(",", "."))
    raise ValueError(f"Unrecognized price format: {raw!r}")


class ReceiptItem(BaseModel):
    product: str = Field(min_length=1)
    price: float = Field(allow_inf_nan=False)
    category: Category

    @field_validator("product", mode="before")
    @classmethod
    def strip_product(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("price", mode="before")
    @classmethod
    def normalize_price(cls, value):
        if isinstance(value, bool):
            raise ValueError("price must be a number")
        if isinstance(value, str):
            return parse_price(value)
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
    items: list[ReceiptItem]


class ReceiptClassificationResponse(ReceiptClassification):
    raw_text: str


class ClassifyTextRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value
