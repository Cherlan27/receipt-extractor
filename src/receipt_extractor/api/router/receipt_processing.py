from fastapi import APIRouter, Depends, UploadFile

from receipt_extractor.api.deps import get_extractor
from receipt_extractor.services.extractor import ReceiptExtractor

process_router = APIRouter()


@process_router.post("/extract_text")
async def extract_text_from_image(
    image: UploadFile, extractor: ReceiptExtractor = Depends(get_extractor)
) -> str:
    image_content = await image.read()
    return extractor.get_text(image_content)
