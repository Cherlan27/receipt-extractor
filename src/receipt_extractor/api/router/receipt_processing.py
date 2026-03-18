from fastapi import APIRouter, Request, UploadFile

process_router = APIRouter()


@process_router.post("/extract_text")
async def extract_text_from_image(request: Request, image: UploadFile) -> str:
    image_content = await image.read()
    extractor = request.app.state.extractor

    text_result = extractor.get_text(image_content)

    return text_result
