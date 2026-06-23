# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with codebase in this repository.

## Project Overview

NeuroView Backend is a FastAPI-based inference server that serves as the backend for the NeuroScan AI desktop application. It accepts raw voxel buffers over a REST API, runs them through PyTorch or TensorFlow/Keras models, and returns segmentation masks as NIfTI files.

The frontend-backend contract is specified in `BACKEND_SPEC.md`. Key points:
- Frontend sends `GET /models` to discover available models, then `POST /predict` with multipart form data
- Input: raw binary voxel buffer (no NIfTI header) + JSON dimensions + model_id
- Output: gzip-compressed NIfTI (`application/octet-stream`) segmentation mask

## Commands

```bash
pip install -r requirements.txt        # Install dependencies
python main.py                          # Start server on port 8000
pytest                                  # Run all tests
pytest tests/test_api_contract.py -k test_models_returns_array_shape  # Run a single test
```

There is no linting or formatting configuration in this project.

## Architecture

Flat, single-module structure. All source files live at the top level:

- **`main.py`** — FastAPI app with all routes, middleware (request ID, rate limiting, CORS), and lifecycle management. Inference is offloaded to a thread pool via `asyncio.to_thread` to avoid blocking the event loop.
- **`engine.py`** — `InferenceEngine` class that scans `models/` at startup, reads `model.json` configs, and loads models using four strategies: TorchScript, PyTorch from source (dynamic import + state dict), Keras SavedModel/H5, or Keras subclassed models (dynamic import + weights). TensorFlow is optionally imported at runtime.
- **`schemas.py`** — Pydantic models: `ModelMetadata` (full model config), `ModelListItem` (frontend-facing), `ErrorResponse`, `VersionResponse`. Enums: `ModelEngine`, `ModelMethod`, `FinalLayer`.
- **`validation.py`** — Pure functions: `parse_dimensions()` (JSON string to 3-tuple), `parse_raw_volume()` (binary bytes to float32 numpy array, trying float32/int16/uint16), `validate_model_id()`.
- **`config.py`** — Frozen `Settings` dataclass driven by environment variables (e.g. `MAX_UPLOAD_BYTES`, `INFERENCE_TIMEOUT_SECONDS`, `ALLOWED_ORIGINS`).

## Model Management

Models are hot-pluggable. Each model lives in its own subdirectory of `models/` with a `model.json` config. Adding a new model requires no code changes — just a new directory with config and weights. See README.md for `model.json` field reference.

## Tests

Tests in `tests/test_api_contract.py` use FastAPI's `TestClient`. They seed the global `engine` singleton with mock models before each test. When adding tests, follow this pattern: create a `_seed_*` helper that populates `engine.loaded_models` and `engine.metadata_store`, then use `TestClient(app)` to hit endpoints.
