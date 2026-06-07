from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from receipt_extractor.api.app import app


@pytest.fixture
def client():
    """TestClient with the heavy ML clients replaced by mocks before lifespan startup runs."""
    with (
        patch("receipt_extractor.api.app.ReceiptExtractor") as mock_extractor_cls,
        patch("receipt_extractor.api.app.LlmClient") as mock_llm_client_cls,
    ):
        mock_extractor_cls.return_value = MagicMock(name="ReceiptExtractor")
        mock_llm_client_cls.return_value = MagicMock(name="LlmClient")

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def mock_extractor(client):
    return client.app.state.extractor


@pytest.fixture
def mock_llm_client(client):
    return client.app.state.llm_client
