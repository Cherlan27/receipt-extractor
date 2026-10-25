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


class ReceiptItem(BaseModel):
    product: str = Field(min_length=1)
    price: float
    category: Category

    @field_validator("price", mode="before")
    @classmethod
    def normalize_price(cls, value):
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d,.\-]", "", value).replace(",", ".")
            return cleaned
        return value

    @field_validator("category", mode="before")
    @classmethod
    def fallback_unknown_category(cls, value):
        try:
            return Category(value)
        except ValueError:
            logger.warning(
                "Unknown category %r, falling back to %s", value, Category.OTHER
            )
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
