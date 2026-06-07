from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from receipt_extractor.api.router.generate import llm_router
from receipt_extractor.api.router.receipt_processing import process_router
from receipt_extractor.llm.client import LlmClient
from receipt_extractor.services.extractor import ReceiptExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.extractor = ReceiptExtractor()
    app.state.llm_client = LlmClient()
    yield
    del app.state.extractor
    del app.state.llm_client


app = FastAPI(title="ReceiptExtractor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(process_router)
app.include_router(llm_router)
