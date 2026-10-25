from fastapi import APIRouter, Depends, HTTPException, UploadFile

from receipt_extractor.api.deps import get_classifier, get_extractor
from receipt_extractor.models.receipt import (
    ClassifyTextRequest,
    ReceiptClassification,
    ReceiptClassificationResponse,
)
from receipt_extractor.services.classifier import ClassificationError, ReceiptClassifier
from receipt_extractor.services.extractor import ReceiptExtractor

classification_router = APIRouter()


def _classify(classifier: ReceiptClassifier, text: str) -> ReceiptClassification:
    try:
        return classifier.classify_text(text)
    except ClassificationError:
        raise HTTPException(
            status_code=502, detail="Classification model returned invalid output"
        )


@classification_router.post("/classify_text")
async def classify_text(
    body: ClassifyTextRequest,
    classifier: ReceiptClassifier = Depends(get_classifier),
) -> ReceiptClassification:
    return _classify(classifier, body.text)


@classification_router.post("/classify_receipt")
async def classify_receipt(
    image: UploadFile,
    extractor: ReceiptExtractor = Depends(get_extractor),
    classifier: ReceiptClassifier = Depends(get_classifier),
) -> ReceiptClassificationResponse:
    text = extractor.get_text(await image.read())
    if not text.strip():
        raise HTTPException(
            status_code=422, detail="No text could be extracted from the image"
        )

    classification = _classify(classifier, text)
    return ReceiptClassificationResponse(items=classification.items, raw_text=text)
