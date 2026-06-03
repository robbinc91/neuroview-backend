# NeuroScan AI Backend Specification

This document defines the backend needed by the NeuroScan AI desktop app for external/cloud segmentation inference.

It is written so you can build a separate backend service that is compatible with the current frontend behavior.

## 1) Scope

The app currently performs most processing locally.  
The backend is required for:

- Discovering available segmentation methods/models from an endpoint entered by user.
- Running remote inference on the loaded volume and returning a segmentation volume.

## 2) Current Frontend Contract

The frontend (Segmentation > External / Cloud Inference) expects:

- Base URL entered by user (example: `http://localhost:8000`)
- `GET {baseUrl}/models`
- `POST {baseUrl}/predict`

### 2.1 `GET /models`

Used to load available methods before inference.

#### Accepted response shape

Response must be a JSON array. Each item may include:

- `id` (string or number) OR `method_id`
- `name` OR `method_name`
- `description` (string recommended)
- `goal` (optional string)
- `input_shape` (optional string or number array)
- `input_format` (optional string, recommended)
- `output_classes` (optional string array)

The app normalizes some aliases (`method_id`, `method_name`) but standard names are preferred.

#### Recommended response example

```json
[
  {
    "id": "unet_brain_v1",
    "name": "UNet Brain Mask",
    "description": "Whole-brain segmentation for T1 MRI volumes.",
    "goal": "Extract intracranial mask",
    "input_format": "Raw float32 voxel stream + dimensions [x,y,z]",
    "input_shape": [256, 256, 160],
    "output_classes": ["0:background", "1:brain"]
  },
  {
    "id": "tumor_multiclass_v2",
    "name": "Tumor Multiclass",
    "description": "Multi-class glioma segmentation.",
    "goal": "Segment edema, non-enhancing core, enhancing tumor",
    "input_format": "Raw float32 voxel stream + dimensions [x,y,z]",
    "output_classes": ["0:bg", "1:edema", "2:core", "3:enhancing"]
  }
]
```

### 2.2 `POST /predict`

Used when user clicks **Run Inference**.

#### Request type

- `multipart/form-data`

#### Form fields expected by app

- `file`: binary blob named `volume.raw`
- `dimensions`: JSON stringified array, e.g. `"[256,256,160]"`
- `model_id`: selected model ID from `/models`

#### Important input notes

- `file` is raw voxel buffer from the current loaded volume (`volume.data.buffer`).
- Voxel numeric type may vary in frontend (`Float32Array`, `Int16Array`, etc.).
- Backend should define strict conversion policy (recommended: convert to `float32` internally).
- No affine/header is sent in this request; only data buffer + dimensions.

#### Response expected by app

- Raw HTTP body as **NIfTI file bytes** (`.nii` or `.nii.gz`) parseable by `nifti-reader-js`.
- App parses response as NIfTI and uses returned volume as segmentation mask.

#### Response requirements

- Must be a valid NIfTI volume.
- Dimensions should match input dimensions.
- Datatype should ideally be `uint8` labels.
- If not `uint8`, app will coerce values to `uint8`.

#### Success response headers (recommended)

- `Content-Type: application/octet-stream`
- `Content-Disposition: attachment; filename="segmentation.nii.gz"`

## 3) Recommended Additional Endpoints

These are not required by the current app, but strongly recommended for production backend:

- `GET /health` -> liveness/readiness
- `GET /version` -> backend version + model pack version
- `GET /models/{id}` -> richer schema for one method
- `POST /predict-json` -> optional JSON-based request for future clients

## 4) Error Handling Contract

### For `GET /models`

- Non-2xx status is treated as connection failure.
- Return useful body for logs, but app currently surfaces status/message.

### For `POST /predict`

Use clear HTTP status codes:

- `400` invalid request (missing fields, invalid dimensions)
- `404` unknown model ID
- `413` payload too large
- `422` input volume/model mismatch
- `500` internal inference failure
- `503` model unavailable/cold start

Recommended JSON error payload:

```json
{
  "error": "Invalid dimensions",
  "detail": "Expected dimensions as JSON array [x,y,z]"
}
```

## 5) CORS and Desktop Behavior

The Electron renderer calls backend via `fetch`, so CORS can still matter depending on runtime config.

Recommended CORS policy:

- Allow origins used by app development and packaged runtime.
- Allow methods: `GET, POST, OPTIONS`
- Allow headers: `Content-Type`
- Allow credentials only if needed.

## 6) Performance Targets

Minimum practical targets for good UX:

- `/models`: < 500 ms typical
- `/predict`: first token/result not applicable; total inference ideally < 10-20 s for common study sizes
- Handle 3D volumes up to at least `512 x 512 x 300` with graceful limits

Recommended backend strategies:

- Async worker queue for long jobs
- Warm model cache
- Optional GPU scheduling
- Max request size limits and memory guards

## 7) Security Requirements

- Validate `model_id` against allow-list.
- Validate dimensions and byte length consistency.
- Reject malformed binary payloads early.
- Add per-IP or per-token rate limiting.
- Add request timeout and inference timeout.
- Log requests with request IDs (without storing PHI unless explicitly required).

## 8) Data/Model Semantics

Frontend segmentation expectations:

- Label `0` = background
- Positive labels = classes/regions
- Multi-class labels supported (e.g. 1, 2, 3, ...)

Returned segmentation should preserve integer labels.

## 9) Backend Tech Recommendations

A practical implementation stack:

- Python + FastAPI (API layer)
- MONAI / PyTorch (model inference)
- nibabel (NIfTI read/write)
- Uvicorn/Gunicorn (serving)
- Optional Redis/Celery for queued jobs

## 10) Reference OpenAPI Draft

```yaml
openapi: 3.0.3
info:
  title: NeuroScan Inference API
  version: 1.0.0
paths:
  /models:
    get:
      summary: List available inference methods
      responses:
        '200':
          description: OK
  /predict:
    post:
      summary: Run segmentation inference
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              required: [file, dimensions, model_id]
              properties:
                file:
                  type: string
                  format: binary
                dimensions:
                  type: string
                  example: "[256,256,160]"
                model_id:
                  type: string
      responses:
        '200':
          description: Segmentation NIfTI bytes
          content:
            application/octet-stream: {}
```

## 11) Compatibility Checklist

Before connecting backend to app, verify:

- [ ] `GET /models` returns non-empty JSON array
- [ ] each model has at least `id` + `name` + `description`
- [ ] app can display input format (`input_format` or `input_shape`)
- [ ] `POST /predict` accepts multipart fields exactly as above
- [ ] response body is valid NIfTI
- [ ] response dimensions equal input dimensions
- [ ] labels are integer-like and usable as segmentation classes

## 12) Suggested Future Enhancements

Not required now, but useful next:

- Add asynchronous job API (`POST /jobs`, `GET /jobs/{id}`)
- Return confidence maps as optional second output
- Support multi-modal inputs (T1/T2/FLAIR) with channel metadata
- Model-specific validation rules in `/models/{id}`

