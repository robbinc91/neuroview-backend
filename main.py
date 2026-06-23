import asyncio
import base64
import json
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
from jobs import cleanup_expired_jobs, create_job, get_job, store_count, update_job
from logger import get_logger
from schemas import (
    ErrorResponse,
    JobCreateRequest,
    JobResponse,
    JobStatus,
    ModelListItem,
    ModelMethod,
    PredictJsonRequest,
    VersionResponse,
)
from validation import parse_dimensions, parse_raw_volume, validate_model_id

log = get_logger(__name__)

# --- LIFECYCLE MANAGEMENT ---
engine = InferenceEngine()
rate_buckets = defaultdict(deque)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 60
_running_jobs = 0
_jobs_lock = asyncio.Lock()
_cleanup_task = None


def _cleanup_rate_buckets():
    now = time.time()
    expired = [
        ip for ip, bucket in rate_buckets.items()
        if not bucket or now - bucket[-1] > RATE_LIMIT_WINDOW_SECONDS * 2
    ]
    for ip in expired:
        del rate_buckets[ip]
    return len(expired)


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(300)
        removed = cleanup_expired_jobs(settings.job_ttl_seconds)
        stale = _cleanup_rate_buckets()
        if removed or stale:
            log.info("Cleaned up %d expired jobs, %d stale rate buckets", removed, stale)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_periodic_cleanup())
    engine.load_all_models()
    log.info("Server started, serving on 0.0.0.0:8000")
    yield
    _cleanup_task.cancel()
    log.info("Server shutting down")

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
        log.warning("Rate limit exceeded for client %s", client_ip)
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

async def _run_job_in_background(
    job_id: str, model_id: str, data: np.ndarray, dims: tuple
):
    global _running_jobs
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(engine.run_inference, model_id, data),
            timeout=settings.inference_timeout_seconds,
        )
        if tuple(result.shape) != tuple(dims):
            update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=time.time(),
                error=f"Expected output dimensions {list(dims)}, got {list(result.shape)}",
            )
            return

        nifti_bytes = _to_nii_gz_bytes(result)
        nifti_b64 = base64.b64encode(nifti_bytes).decode("ascii")
        update_job(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=time.time(),
            result_nifti_base64=nifti_b64,
        )
    except asyncio.TimeoutError:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=time.time(),
            error="Inference timed out",
        )
    except Exception as exc:
        update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=time.time(),
            error=str(exc),
        )
    finally:
        async with _jobs_lock:
            _running_jobs -= 1


def _decode_and_validate(body, request) -> tuple | None:
    """Shared validation for JSON endpoints. Returns (dims, data) or sends an error response."""
    if body.model_id not in engine.loaded_models:
        return _error(404, "Unknown model", f"Model '{body.model_id}' not found", request)

    meta = engine.metadata_store[body.model_id]
    if meta.method != ModelMethod.SEGMENTATION:
        return _error(422, "Input/model mismatch", "Selected model is not a segmentation model", request)

    try:
        dims = parse_dimensions(json.dumps(body.dimensions))
    except ValueError as exc:
        return _error(400, "Invalid dimensions", str(exc), request)

    try:
        content = base64.b64decode(body.file_base64)
    except Exception as exc:
        return _error(400, "Invalid base64 payload", str(exc), request)

    if len(content) > settings.max_upload_bytes:
        return _error(413, "Payload too large", "Input exceeds configured maximum size", request)

    try:
        data = parse_raw_volume(content, dims, settings.max_voxels)
    except ValueError as exc:
        return _error(400, "Invalid payload", str(exc), request)

    return dims, data


@app.post("/predict-json")
async def predict_json(
    request: Request,
    body: PredictJsonRequest,
):
    """JSON-based inference endpoint — returns NIfTI bytes synchronously."""
    result = _decode_and_validate(body, request)
    if isinstance(result, JSONResponse):
        return result
    dims, data = result

    try:
        inference_result = await asyncio.wait_for(
            asyncio.to_thread(engine.run_inference, body.model_id, data),
            timeout=settings.inference_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return _error(503, "Model unavailable", "Inference timed out", request)
    except Exception as exc:
        return _error(500, "Inference failed", str(exc), request)

    if tuple(inference_result.shape) != tuple(dims):
        return _error(
            422,
            "Input/model mismatch",
            f"Expected output dimensions {list(dims)}, got {list(inference_result.shape)}",
            request,
        )

    payload = _to_nii_gz_bytes(inference_result)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="segmentation.nii.gz"'},
    )


@app.post("/jobs")
async def submit_job(
    request: Request,
    body: JobCreateRequest,
):
    """Submit an inference job — returns immediately with a job_id."""
    global _running_jobs

    result = _decode_and_validate(body, request)
    if isinstance(result, JSONResponse):
        return result
    dims, data = result

    async with _jobs_lock:
        if _running_jobs >= settings.max_concurrent_jobs:
            return _error(503, "Too many jobs", "Maximum concurrent jobs reached", request)
        _running_jobs += 1

    record = create_job(body.model_id)
    update_job(record.job_id, status=JobStatus.RUNNING, started_at=time.time())

    asyncio.create_task(_run_job_in_background(record.job_id, body.model_id, data, dims))

    return JobResponse(
        job_id=record.job_id,
        status=JobStatus.RUNNING,
        model_id=body.model_id,
        created_at=record.created_at,
        started_at=record.started_at,
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request):
    """Poll a job for status and result."""
    record = get_job(job_id)
    if record is None:
        return _error(404, "Job not found", f"Job '{job_id}' not found", request)

    if store_count() > 100:
        cleanup_expired_jobs(settings.job_ttl_seconds)

    return JobResponse(
        job_id=record.job_id,
        status=record.status,
        model_id=record.model_id,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        result_nifti_base64=record.result_nifti_base64,
        error=record.error,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)