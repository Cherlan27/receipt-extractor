import io

import pytest

from receipt_extractor.models.receipt import ReceiptClassification
from receipt_extractor.services.classifier import ClassificationError

CLASSIFICATION = ReceiptClassification.model_validate(
    {
        "items": [
            {"product": "Apfel", "price": 1.99, "category": "Obst"},
            {"product": "Gummibärchen", "price": "0,99", "category": "Süßigkeiten"},
        ]
    }
)
EXPECTED_ITEMS = [
    {"product": "Apfel", "price": 1.99, "category": "Obst"},
    {"product": "Gummibärchen", "price": 0.99, "category": "Süßigkeiten"},
]


def _upload(content: bytes = b"fake-image-bytes"):
    return {"image": ("receipt.png", io.BytesIO(content), "image/png")}


class TestClassifyTextEndpoint:
    def test_returns_classified_items(self, client, mock_classifier):
        mock_classifier.classify_text.return_value = CLASSIFICATION

        response = client.post(
            "/classify_text", json={"text": "Apfel 1.99\nGummibärchen 0,99"}
        )

        assert response.status_code == 200
        assert response.json() == {"items": EXPECTED_ITEMS}
        mock_classifier.classify_text.assert_called_once_with(
            "Apfel 1.99\nGummibärchen 0,99"
        )

    def test_does_not_invoke_ocr(self, client, mock_classifier, mock_extractor):
        mock_classifier.classify_text.return_value = CLASSIFICATION

        client.post("/classify_text", json={"text": "Apfel 1.99"})

        mock_extractor.get_text.assert_not_called()

    def test_returns_empty_items_for_empty_classification(
        self, client, mock_classifier
    ):
        mock_classifier.classify_text.return_value = ReceiptClassification(items=[])

        response = client.post("/classify_text", json={"text": "nothing"})

        assert response.status_code == 200
        assert response.json() == {"items": []}

    @pytest.mark.parametrize("body", [{"text": ""}, {"text": "   "}, {}])
    def test_returns_422_for_empty_or_missing_text(self, client, mock_classifier, body):
        response = client.post("/classify_text", json=body)

        assert response.status_code == 422
        mock_classifier.classify_text.assert_not_called()

    def test_returns_502_when_model_output_is_invalid(self, client, mock_classifier):
        mock_classifier.classify_text.side_effect = ClassificationError(
            "bad json", "Hier ist deine Liste"
        )

        response = client.post("/classify_text", json={"text": "Apfel 1.99"})

        assert response.status_code == 502
        assert response.json() == {
            "detail": "Classification model returned invalid output"
        }
        assert "Hier ist deine Liste" not in response.text


class TestClassifyReceiptEndpoint:
    def test_returns_items_and_raw_text(self, client, mock_extractor, mock_classifier):
        mock_extractor.get_text.return_value = "Apfel 1.99\nGummibärchen 0,99"
        mock_classifier.classify_text.return_value = CLASSIFICATION

        response = client.post("/classify_receipt", files=_upload())

        assert response.status_code == 200
        assert response.json() == {
            "items": EXPECTED_ITEMS,
            "raw_text": "Apfel 1.99\nGummibärchen 0,99",
        }
        mock_extractor.get_text.assert_called_once_with(b"fake-image-bytes")
        mock_classifier.classify_text.assert_called_once_with(
            "Apfel 1.99\nGummibärchen 0,99"
        )

    @pytest.mark.parametrize("ocr_text", ["", "   \n"])
    def test_returns_422_when_ocr_yields_no_text(
        self, client, mock_extractor, mock_classifier, ocr_text
    ):
        mock_extractor.get_text.return_value = ocr_text

        response = client.post("/classify_receipt", files=_upload())

        assert response.status_code == 422
        assert response.json() == {
            "detail": "No text could be extracted from the image"
        }
        mock_classifier.classify_text.assert_not_called()

    def test_returns_422_when_no_image_is_provided(
        self, client, mock_extractor, mock_classifier
    ):
        response = client.post("/classify_receipt")

        assert response.status_code == 422
        mock_extractor.get_text.assert_not_called()
        mock_classifier.classify_text.assert_not_called()

    def test_returns_502_when_model_output_is_invalid(
        self, client, mock_extractor, mock_classifier
    ):
        mock_extractor.get_text.return_value = "Apfel 1.99"
        mock_classifier.classify_text.side_effect = ClassificationError("bad", "prose")

        response = client.post("/classify_receipt", files=_upload())

        assert response.status_code == 502
        assert response.json() == {
            "detail": "Classification model returned invalid output"
        }
