from fastapi import APIRouter, Depends

from receipt_extractor.api.deps import get_llm_client
from receipt_extractor.llm.client import LlmClient
from receipt_extractor.models.llm_message import PromptData

llm_router = APIRouter(prefix="/llm")


@llm_router.post("/generate")
async def generate(
    body: PromptData, llm_client: LlmClient = Depends(get_llm_client)
) -> dict:
    return llm_client.generate(body)
