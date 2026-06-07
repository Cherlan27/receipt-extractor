# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Receipt Extractor is a FastAPI-based service that extracts text from receipt images using the GLM-OCR model from HuggingFace, and additionally exposes a local LLM chat-completion endpoint. It uses PyTorch and transformer models to perform OCR on receipt images and to run a separate causal-LM for general text generation.

## Setup and Dependencies

### Package Manager
This project uses **Poetry** for dependency management (`poetry@2.3.2` or later). The project requires **Python 3.12+** (up to <3.15).

### Installation
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Key Dependencies
- **FastAPI & Uvicorn**: REST API framework and server (fastapi@0.135, uvicorn@0.42)
- **PyTorch + CUDA**: Deep learning framework with GPU support (torch@2.10, torchvision@0.25, torchaudio@2.10)
  - PyTorch wheels are sourced from `https://download.pytorch.org/whl/cu128` (CUDA 12.8 GPU support)
- **Transformers**: HuggingFace model loading and inference (from main branch)
- **Pillow**: Image processing (pillow@12.1)
- **Accelerate**: GPU acceleration (accelerate@1.12)
- **load-dotenv**: Loads `.env` into the environment (used by `LlmClient` to read `LLM_MODEL_PATH`)

### Development Dependencies
- **Pre-commit**: Code quality hooks (ruff check/format, isort, trailing-whitespace, merge-conflict detection)
- **IPykernel**: Jupyter notebook kernel

## Commands

### Running the Application
```bash
# Start the FastAPI server (runs on http://0.0.0.0:8080)
poetry run python -m src.receipt_extractor.main

# From within the src/receipt_extractor directory:
cd src/receipt_extractor && python main.py
```

### Code Quality

```bash
# Run pre-commit hooks (ruff check/format, isort, trailing-whitespace, etc.)
pre-commit run --all-files

# Run ruff linter/formatter
poetry run ruff check --fix src/
poetry run ruff format src/

# Run isort for import sorting
poetry run isort src/
```

### Testing
Currently, there are minimal tests in the `tests/` directory. No test runner is configured. Consider adding pytest for automated testing of the `ReceiptExtractor`/`LlmClient` classes and API endpoints.

## Architecture

### High-Level Flow
1. **API Layer** (`src/receipt_extractor/api/`): FastAPI application with CORS middleware
2. **Dependency Injection** (`src/receipt_extractor/api/deps.py`): FastAPI `Depends` providers that pull shared singletons (`ReceiptExtractor`, `LlmClient`) off `request.app.state`
3. **Routing** (`src/receipt_extractor/api/router/`): one router module per feature area, each declaring its own `APIRouter` and included in `app.py`
4. **Services / LLM clients** (`src/receipt_extractor/services/`, `src/receipt_extractor/llm/`): model-loading and inference logic, kept independent of FastAPI
5. **Models** (`src/receipt_extractor/models/`): Pydantic request/response schemas shared between routers and clients
6. **Main Entry Point** (`src/receipt_extractor/main.py`): Starts Uvicorn server with the FastAPI app

### Key Components

#### ReceiptExtractor (`src/receipt_extractor/services/extractor.py`)
- **Purpose**: Loads and manages the GLM-OCR model, performs text extraction from images
- **Model**: `zai-org/GLM-OCR` (image-to-text model from HuggingFace), loaded with `device_map="auto"`
- **Key Methods**:
  - `__init__()`: Loads the model and processor on initialization
  - `get_text(image: bytes) -> str`: Takes raw image bytes, builds a chat-template prompt (`{"type": "image", ...}` + OCR instruction), generates up to 8192 tokens, and returns the decoded text
    - Returns `""` on `UnidentifiedImageError`/unexpected errors instead of raising

#### LlmClient (`src/receipt_extractor/llm/client.py`)
- **Purpose**: Loads a separate local causal-LM (e.g. an OpenChat-style model) and runs chat-style generation independent of the OCR pipeline
- **Model path**: `LLM_MODEL_PATH` env var (loaded via `load_dotenv()`), defaulting to `./openchat_model`; loaded with `torch_dtype=torch.float16` onto CUDA if available, else CPU
- **Key Methods**:
  - `generate(req: PromptData) -> dict`: Prepends a fixed `system_prompt` (currently hardcoded to "Answer in French" — check before relying on it for other languages), applies the tokenizer's chat template, generates `req.max_new_tokens` tokens, and returns `{"response": <decoded text>}`

#### Pydantic Schemas (`src/receipt_extractor/models/llm_message.py`)
- `ChatMessage`: `{role: str, content: str}`
- `PromptData`: `{messages: list[ChatMessage], max_new_tokens: int = 1000}` — request body for `/llm/generate`
- `Word`, `TopicRequest`: additional schemas not yet wired up to any endpoint

#### FastAPI App (`src/receipt_extractor/api/app.py`)
- **Lifespan Management**: `ReceiptExtractor` and `LlmClient` are each instantiated once on startup (loading their models) and shared across requests via `app.state.extractor` / `app.state.llm_client`, then torn down on shutdown
- **CORS**: Allows all origins, credentials, methods, and headers
- **Routers**: Includes `process_router` (OCR) and `llm_router` (LLM generation)

#### Routers (`src/receipt_extractor/api/router/`)
- `receipt_processing.py` — `process_router`: **POST /extract_text** accepts a multipart image upload, reads the bytes, and delegates to `ReceiptExtractor.get_text()` via `Depends(get_extractor)`
- `generate.py` — `llm_router` (prefix `/llm`): **POST /llm/generate** accepts a `PromptData` body and delegates to `LlmClient.generate()` via `Depends(get_llm_client)`

#### Logging (`src/receipt_extractor/logging_config.py`)
- Simple INFO-level logging with timestamp, level, logger name, and message

### Data Flow

OCR pipeline:
```
Client Request (image file)
        ↓
    process_router (POST /extract_text)
        ↓
    ReceiptExtractor.get_text()
        ↓
    GLM-OCR Model (HuggingFace)
        ↓
    Extracted Text (string) → API Response
```

LLM generation pipeline:
```
Client Request (PromptData: messages + max_new_tokens)
        ↓
    llm_router (POST /llm/generate)
        ↓
    LlmClient.generate()
        ↓
    Local causal-LM (path from LLM_MODEL_PATH)
        ↓
    {"response": <generated text>} → API Response
```

### Configuration
- Environment variables are defined in `.env` and loaded via `load_dotenv()`:
  - `USERNAME`, `PASSWORD`, `DOWNLOAD_FOALDER` — for the IMAP/email-download integration (see `notebooks/read_mails.ipynb`); not yet wired into the API
  - `LLM_MODEL_PATH` — filesystem path to the local causal-LM used by `LlmClient`
- Debug configuration available in `.vscode/launch.json` for debugpy

### Work in progress / stub modules
These exist on disk but are currently empty or reference modules that don't exist yet — don't assume they're functional without checking first:
- `src/receipt_extractor/services/classifier.py` — empty
- `src/receipt_extractor/config.py` — empty
- `src/receipt_extractor/services/__init__db.py` — imports `receipt_extractor.services.database` (Base/engine), which does not currently exist in the tree
- `src/receipt_extractor/utils/` — empty directory

## Notes for Development

### GPU Support
The project is configured for CUDA 12.8. Verify GPU availability with the `ReceiptExtractor.get_gpu_state()` static method (currently prints to stdout). Update PyTorch wheel source in `pyproject.toml` if using different CUDA version. `LlmClient` separately checks `torch.cuda.is_available()` and falls back to CPU.

### Model Loading
Both the GLM-OCR model and the local LLM are loaded once at FastAPI startup (in `lifespan`), so the first request after boot doesn't pay the load cost — but server startup itself will be slow and requires the model weights to be available (HuggingFace cache for GLM-OCR, local path at `LLM_MODEL_PATH` for the LLM).

### Notebooks
- `src/receipt_extractor/notebooks/image_to_text.ipynb` — experimental testing of the OCR pipeline
- `src/receipt_extractor/notebooks/read_mails.ipynb` — experimental IMAP email reading (related to the `USERNAME`/`PASSWORD`/`DOWNLOAD_FOALDER` env vars)

### Testing Data
Sample receipt images are in `data/` directory (aldi_quittung.jpeg, test.jpg, etc.) for manual testing of the extraction service.
