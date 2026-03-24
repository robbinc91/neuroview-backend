import asyncio
import time
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
from uuid import uuid4
from collections import defaultdict, deque

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from config import settings
from engine import InferenceEngine
from schemas import ErrorResponse, ModelListItem, ModelMethod, VersionResponse
from validation import parse_dimensions, parse_raw_volume, validate_model_id

# --- LIFECYCLE MANAGEMENT ---
engine = InferenceEngine()
rate_buckets = defaultdict(deque)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 60

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    engine.load_all_models()
    yield
    # Shutdown logic if needed

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = rate_buckets[client_ip]
    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
        payload = ErrorResponse(
            error="Rate limit exceeded",
            detail="Too many requests, please retry later",
            request_id=str(uuid4()),
        )
        return JSONResponse(status_code=429, content=payload.model_dump())
    bucket.append(now)
    return await call_next(request)


def _error(status_code: int, error: str, detail: str, request: Request) -> JSONResponse:
    payload = ErrorResponse(error=error, detail=detail, request_id=request.state.request_id)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _serialize_models() -> list[dict]:
    items = []
    for model_id, meta in engine.metadata_store.items():
        display_name = meta.name or model_id
        description = meta.description or f"{meta.method.value.title()} model using {meta.engine.value}"
        model_item = ModelListItem(
            id=model_id,
            name=display_name,
            description=description,
            goal=meta.goal,
            input_shape=meta.input_shape,
            input_format=meta.input_format or "Raw float32 voxel stream + dimensions [x,y,z]",
            output_classes=meta.output_classes,
            method_id=model_id,
            method_name=display_name,
        )
        items.append(model_item.model_dump())
    return items


def _to_nii_gz_bytes(volume: np.ndarray) -> bytes:
    image = nib.Nifti1Image(volume.astype(np.uint8, copy=False), np.eye(4))
    with NamedTemporaryFile(suffix=".nii.gz", delete=True) as tmp:
        nib.save(image, tmp.name)
        tmp.seek(0)
        return tmp.read()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error(400, "Invalid request", str(exc), request)


@app.get("/models")
def list_models():
    """Return available models in frontend-compatible array shape."""
    return _serialize_models()

@app.get("/models/{model_id}")
def get_model(model_id: str, request: Request):
    if model_id not in engine.metadata_store:
        return _error(404, "Unknown model", f"Model '{model_id}' not found", request)
    for item in _serialize_models():
        if str(item["id"]) == model_id:
            return item
    return _error(404, "Unknown model", f"Model '{model_id}' not found", request)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": len(engine.metadata_store)}


@app.get("/version", response_model=VersionResponse)
def version():
    return VersionResponse(version=settings.backend_version, model_pack_version=settings.model_pack_version)


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    dimensions: str = Form(...),
    model_id: str = Form(...),
):
    if model_id not in engine.loaded_models:
        return _error(404, "Unknown model", f"Model '{model_id}' not found", request)
    if len(engine.loaded_models) == 0:
        return _error(503, "Model unavailable", "No models are loaded", request)

    try:
        validate_model_id(model_id, engine.loaded_models.keys())
    except KeyError:
        return _error(404, "Unknown model", f"Model '{model_id}' not found", request)
    meta = engine.metadata_store[model_id]
    if meta.method != ModelMethod.SEGMENTATION:
        return _error(422, "Input/model mismatch", "Selected model is not a segmentation model", request)

    try:
        dims = parse_dimensions(dimensions)
    except ValueError as exc:
        return _error(400, "Invalid dimensions", str(exc), request)

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        return _error(413, "Payload too large", "Input file exceeds configured maximum size", request)

    try:
        data = parse_raw_volume(content, dims, settings.max_voxels)
    except ValueError as exc:
        return _error(400, "Invalid payload", str(exc), request)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(engine.run_inference, model_id, data),
            timeout=settings.inference_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return _error(503, "Model unavailable", "Inference timed out", request)
    except Exception as exc:
        return _error(500, "Inference failed", str(exc), request)

    if tuple(result.shape) != tuple(dims):
        return _error(
            422,
            "Input/model mismatch",
            f"Expected output dimensions {list(dims)}, got {list(result.shape)}",
            request,
        )

    payload = _to_nii_gz_bytes(result)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="segmentation.nii.gz"'},
    )


@app.post("/predict/{model_id}")
async def predict_legacy(model_id: str, file: UploadFile = File(...)):
    """
    Legacy endpoint retained for compatibility.
    This path accepts a raw voxel upload only for fallback clients.
    """
    _ = model_id
    _ = await file.read()
    payload = ErrorResponse(
        error="Invalid request",
        detail="Use POST /predict with multipart fields: file, dimensions, model_id",
        request_id=str(uuid4()),
    )
    return JSONResponse(
        status_code=400,
        content=payload.model_dump(),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)