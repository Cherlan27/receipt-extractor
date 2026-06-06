# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Receipt Extractor is a FastAPI-based service that extracts text from receipt images using the GLM-OCR model from HuggingFace. It uses PyTorch and transformer models to perform optical character recognition on receipt images and return extracted text.

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
Currently, there are minimal tests in the `tests/` directory. No test runner is configured. Consider adding pytest for automated testing of the ReceiptExtractor class and API endpoints.

## Architecture

### High-Level Flow
1. **API Layer** (`src/receipt_extractor/api/`): FastAPI application with CORS middleware
2. **Routing** (`src/receipt_extractor/api/router/receipt_processing.py`): Handles `POST /extract_text` endpoint
3. **Services** (`src/receipt_extractor/services/`): Core business logic for image processing
4. **Main Entry Point** (`src/receipt_extractor/main.py`): Starts Uvicorn server with FastAPI app

### Key Components

#### ReceiptExtractor Class (`src/receipt_extractor/services/extractor.py`)
- **Purpose**: Loads and manages the GLM-OCR model, performs text extraction from images
- **Model**: `zai-org/GLM-OCR` (image-to-text model from HuggingFace)
- **Key Methods**:
  - `__init__()`: Loads the model and processor on initialization with auto device mapping
  - `get_text(image: bytes) -> str`: Takes raw image bytes, returns extracted text
    - Handles image loading errors gracefully
    - Uses the processor's chat template API to prepare inputs
    - Generates up to 8192 tokens of output
    - Returns empty string on failure

#### FastAPI App (`src/receipt_extractor/api/app.py`)
- **Lifespan Management**: ReceiptExtractor instance is created once on startup and shared across requests via `app.state.extractor`
- **CORS**: Allows all origins, credentials, methods, and headers
- **Router**: Includes `process_router` with receipt extraction endpoint

#### API Router (`src/receipt_extractor/api/router/receipt_processing.py`)
- **POST /extract_text**: Accepts multipart file upload, returns extracted text string
  - Reads image from request, delegates to extractor service

#### Logging (`src/receipt_extractor/logging_config.py`)
- Simple INFO-level logging with timestamp, level, logger name, and message

### Data Flow
```
Client Request (image file)
        ↓
    Router (/extract_text)
        ↓
    ReceiptExtractor.get_text()
        ↓
    GLM-OCR Model (HuggingFace)
        ↓
    Extracted Text (string)
        ↓
    API Response
```

### Configuration
- No config files currently used. Environment variables are defined in `.env`:
  - `USERNAME`, `PASSWORD`, `DOWNLOAD_FOALDER` (for external integrations, not yet implemented)
- Debug configuration available in `.vscode/launch.json` for debugpy

## Notes for Development

### GPU Support
The project is configured for CUDA 12.8. Verify GPU availability with the `ReceiptExtractor.get_gpu_state()` static method (currently prints to stdout). Update PyTorch wheel source in `pyproject.toml` if using different CUDA version.

### Model Loading
The GLM-OCR model is downloaded from HuggingFace on first instantiation. Large model weights will be cached locally. First run will require internet connectivity and may take time.

### Notebooks
A Jupyter notebook exists at `src/receipt_extractor/notebooks/image_to_text.ipynb` for experimental testing of the OCR pipeline.

### Testing Data
Sample receipt images are in `data/` directory (aldi_quittung.jpeg, test.jpg, etc.) for manual testing of the extraction service.
