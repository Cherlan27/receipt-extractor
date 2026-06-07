from receipt_extractor.models.llm_message import ChatMessage


class TestGenerateEndpoint:
    def test_returns_llm_response_for_valid_prompt(self, client, mock_llm_client):
        mock_llm_client.generate.return_value = {"response": "Bonjour le monde"}

        response = client.post(
            "/llm/generate",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "max_new_tokens": 50,
            },
        )

        assert response.status_code == 200
        assert response.json() == {"response": "Bonjour le monde"}

        (called_request,), _ = mock_llm_client.generate.call_args
        assert called_request.messages == [ChatMessage(role="user", content="Hello")]
        assert called_request.max_new_tokens == 50

    def test_uses_default_max_new_tokens_when_omitted(self, client, mock_llm_client):
        mock_llm_client.generate.return_value = {"response": "ok"}

        response = client.post(
            "/llm/generate", json={"messages": [{"role": "user", "content": "Hi"}]}
        )

        assert response.status_code == 200
        (called_request,), _ = mock_llm_client.generate.call_args
        assert called_request.max_new_tokens == 1000

    def test_returns_422_for_missing_messages(self, client, mock_llm_client):
        response = client.post("/llm/generate", json={})

        assert response.status_code == 422
        mock_llm_client.generate.assert_not_called()
