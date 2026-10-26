import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from receipt_extractor.services.extractor import (
    MODEL_PATH,
    ReceiptExtractor,
    strip_special_tokens,
)


class _FakeBatchEncoding(dict):
    """Minimal stand-in for transformers' BatchEncoding: a dict with `.to()`."""

    def to(self, device):
        return self


def _sample_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def extractor():
    with (
        patch(
            "receipt_extractor.services.extractor.AutoProcessor"
        ) as mock_processor_cls,
        patch(
            "receipt_extractor.services.extractor.AutoModelForImageTextToText"
        ) as mock_model_cls,
    ):
        mock_processor_cls.from_pretrained.return_value = MagicMock(name="processor")
        mock_model_cls.from_pretrained.return_value = MagicMock(name="model")

        yield ReceiptExtractor()


class TestInit:
    def test_loads_processor_and_model_from_model_path(self):
        with (
            patch(
                "receipt_extractor.services.extractor.AutoProcessor"
            ) as mock_processor_cls,
            patch(
                "receipt_extractor.services.extractor.AutoModelForImageTextToText"
            ) as mock_model_cls,
        ):
            instance = ReceiptExtractor(model_path="custom/model-path")

            assert instance.model_path == "custom/model-path"
            mock_processor_cls.from_pretrained.assert_called_once_with(
                "custom/model-path"
            )
            mock_model_cls.from_pretrained.assert_called_once_with(
                pretrained_model_name_or_path="custom/model-path",
                torch_dtype="auto",
                device_map="auto",
            )


class TestGetText:
    def test_returns_decoded_text_for_valid_image(self, extractor):
        inputs = _FakeBatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "token_type_ids": torch.tensor([[0, 0, 0, 0, 0]]),
            }
        )
        extractor.processor.apply_chat_template.return_value = inputs
        extractor.model.generate.return_value = [torch.tensor([1, 2, 3, 4, 5, 6, 7])]
        extractor.processor.decode.return_value = "EXTRACTED RECEIPT TEXT"

        result = extractor.get_text(_sample_image_bytes())

        assert result == "EXTRACTED RECEIPT TEXT"

        extractor.model.generate.assert_called_once()
        _, generate_kwargs = extractor.model.generate.call_args
        assert generate_kwargs["max_new_tokens"] == 8192
        assert "token_type_ids" not in generate_kwargs

        decoded_ids, decode_kwargs = extractor.processor.decode.call_args
        assert torch.equal(decoded_ids[0], torch.tensor([6, 7]))
        assert decode_kwargs["skip_special_tokens"] is True

    def test_returns_decoded_text_verbatim(self, extractor):
        receipt_text = "ALDI\nBIO TOMATENMARK\n1,15 € 1\n\nZU ZAHLEN\n21,89 €"
        self._prepare_generation(extractor, decoded=receipt_text)

        assert extractor.get_text(_sample_image_bytes()) == receipt_text

    def test_returns_empty_string_when_output_has_only_control_tokens(self, extractor):
        # With special tokens skipped, an output of only "<|user|>" decodes to "".
        self._prepare_generation(extractor, decoded="")

        assert extractor.get_text(_sample_image_bytes()) == ""

    def test_strips_special_tokens_that_decode_still_emits(self, extractor):
        # <|begin_of_box|>/<|end_of_box|> survive skip_special_tokens=True on GLM-OCR
        extractor.processor.tokenizer.all_special_tokens = [
            "<|user|>",
            "<|begin_of_box|>",
            "<|end_of_box|>",
        ]
        self._prepare_generation(
            extractor, decoded="<|begin_of_box|>ALDI\n1,99 €<|end_of_box|><|user|>"
        )

        assert extractor.get_text(_sample_image_bytes()) == "ALDI\n1,99 €"

    def test_returns_empty_string_for_truncated_image_with_valid_header(
        self, extractor
    ):
        # Image.open only reads the header; the corrupt body must still be caught.
        truncated = _sample_image_bytes()[:-30]

        assert extractor.get_text(truncated) == ""
        extractor.model.generate.assert_not_called()

    @staticmethod
    def _prepare_generation(extractor, decoded):
        extractor.processor.apply_chat_template.return_value = _FakeBatchEncoding(
            {"input_ids": torch.tensor([[1, 2, 3]])}
        )
        extractor.model.generate.return_value = [torch.tensor([1, 2, 3, 4])]
        extractor.processor.decode.return_value = decoded

    def test_returns_empty_string_for_unparsable_image_bytes(self, extractor):
        result = extractor.get_text(b"this is not an image")

        assert result == ""
        extractor.model.generate.assert_not_called()

    def test_returns_empty_string_on_unexpected_processing_error(self, extractor):
        with patch(
            "receipt_extractor.services.extractor.Image.open",
            side_effect=RuntimeError("boom"),
        ):
            result = extractor.get_text(_sample_image_bytes())

        assert result == ""
        extractor.model.generate.assert_not_called()


class TestStripSpecialTokens:
    def test_removes_every_listed_token(self):
        tokenizer = SimpleNamespace(
            all_special_tokens=["<|user|>", "<|begin_of_box|>", "<|end_of_box|>"]
        )

        text = "x<|begin_of_box|>y<|end_of_box|>z<|user|>"

        assert strip_special_tokens(text, tokenizer) == "xyz"

    def test_longer_tokens_win_over_their_prefixes(self):
        tokenizer = SimpleNamespace(all_special_tokens=["<a>", "<a>b"])

        assert strip_special_tokens("1<a>b2", tokenizer) == "12"

    def test_leaves_ordinary_text_untouched(self):
        tokenizer = SimpleNamespace(all_special_tokens=["<|user|>"])
        text = "Preis < 5 | Menge > 2\n1,99 €"

        assert strip_special_tokens(text, tokenizer) == text


class TestWithRealGlmOcrTokenizer:
    """Pins the tokenizer behavior the extractor relies on; skipped if not cached."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        except Exception as e:
            pytest.skip(f"GLM-OCR tokenizer is not available locally: {e}")

    @pytest.mark.parametrize(
        "token",
        ["<|user|>", "<|endoftext|>", "<|begin_of_box|>", "<|end_of_box|>"],
    )
    def test_no_special_token_survives_decode_plus_strip(self, tokenizer, token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_ids = tokenizer.encode("ALDI 1,99", add_special_tokens=False)

        decoded = tokenizer.decode([*text_ids, token_id], skip_special_tokens=True)

        assert strip_special_tokens(decoded, tokenizer) == "ALDI 1,99"
