from contextlib import asynccontextmanager

from api.router.receipt_processing import process_router
from fastapi import FastAPI
from services.extractor import ReceiptExtractor
from starlette.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.extractor = ReceiptExtractor()
    yield
    del app.state.extractor


app = FastAPI(title="ReceiptExtractor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(process_router)
