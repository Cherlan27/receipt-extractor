from fastapi import Request

from receipt_extractor.llm.client import LlmClient
from receipt_extractor.services.extractor import ReceiptExtractor


def get_llm_client(request: Request) -> LlmClient:
    return request.app.state.llm_client


def get_extractor(request: Request) -> ReceiptExtractor:
    return request.app.state.extractor
