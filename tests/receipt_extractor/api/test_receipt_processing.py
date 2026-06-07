import io


class TestExtractTextEndpoint:
    def test_returns_extracted_text_for_uploaded_image(self, client, mock_extractor):
        mock_extractor.get_text.return_value = "TOTAL: 12.34 EUR"

        response = client.post(
            "/extract_text",
            files={
                "image": ("receipt.png", io.BytesIO(b"fake-image-bytes"), "image/png")
            },
        )

        assert response.status_code == 200
        assert response.json() == "TOTAL: 12.34 EUR"
        mock_extractor.get_text.assert_called_once_with(b"fake-image-bytes")

    def test_returns_422_when_no_image_is_provided(self, client, mock_extractor):
        response = client.post("/extract_text")

        assert response.status_code == 422
        mock_extractor.get_text.assert_not_called()
