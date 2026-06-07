import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from receipt_extractor.services.extractor import ReceiptExtractor


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
        assert decode_kwargs["skip_special_tokens"] is False

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
