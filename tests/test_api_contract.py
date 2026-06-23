import base64
import gzip
import io
import json
import time

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from main import app, engine
from schemas import ModelEngine, ModelMetadata, ModelMethod


def _seed_segmentation_model():
    model_id = "test_seg_model"
    engine.loaded_models = {model_id: object()}
    engine.metadata_store = {
        model_id: ModelMetadata(
            id=model_id,
            checkpoint_name="mock.pt",
            method=ModelMethod.SEGMENTATION,
            engine=ModelEngine.TORCH,
            name="Test Seg Model",
            description="Test segmentation model",
            output_classes=["0:bg", "1:fg"],
        )
    }
    return model_id


def test_models_returns_array_shape():
    _seed_segmentation_model()
    client = TestClient(app)

    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["id"] == "test_seg_model"
    assert "name" in body[0]
    assert "description" in body[0]


def test_predict_accepts_raw_and_returns_nifti():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    dims = [8, 8, 4]
    volume = np.zeros(dims, dtype=np.float32)
    volume[1:3, 1:3, 1:2] = 1.0
    payload = volume.tobytes()

    def _fake_run_inference(_model_id, input_data):
        assert _model_id == model_id
        assert tuple(input_data.shape) == tuple(dims)
        return (input_data > 0.5).astype(np.uint8)

    engine.run_inference = _fake_run_inference

    response = client.post(
        "/predict",
        files={"file": ("volume.raw", payload, "application/octet-stream")},
        data={"dimensions": json.dumps(dims), "model_id": model_id},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    out_bytes = response.content

    with io.BytesIO(out_bytes) as bio:
        with gzip.GzipFile(fileobj=bio) as gz:
            nii_bytes = gz.read()
    img = nib.Nifti1Image.from_bytes(nii_bytes)
    assert tuple(img.shape) == tuple(dims)


def test_predict_rejects_bad_dimensions():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    payload = (np.zeros((4, 4, 4), dtype=np.float32)).tobytes()
    response = client.post(
        "/predict",
        files={"file": ("volume.raw", payload, "application/octet-stream")},
        data={"dimensions": "[4,4]", "model_id": model_id},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "Invalid dimensions"


# ---- predict-json tests ----

def _make_b64_volume(dims):
    volume = np.zeros(dims, dtype=np.float32)
    volume[1:3, 1:3, 1:2] = 1.0
    return base64.b64encode(volume.tobytes()).decode("ascii")


def test_predict_json_returns_nifti():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    dims = [8, 8, 4]

    def _fake(_mid, input_data):
        return (input_data > 0.5).astype(np.uint8)

    engine.run_inference = _fake

    response = client.post(
        "/predict-json",
        json={
            "file_base64": _make_b64_volume(dims),
            "dimensions": dims,
            "model_id": model_id,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    out_bytes = response.content
    with io.BytesIO(out_bytes) as bio:
        with gzip.GzipFile(fileobj=bio) as gz:
            nii_bytes = gz.read()
    img = nib.Nifti1Image.from_bytes(nii_bytes)
    assert tuple(img.shape) == tuple(dims)


def test_predict_json_rejects_bad_base64():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    response = client.post(
        "/predict-json",
        json={
            "file_base64": "not-valid-base64!!",
            "dimensions": [8, 8, 4],
            "model_id": model_id,
        },
    )
    assert response.status_code == 400


def test_predict_json_rejects_bad_dimensions():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    response = client.post(
        "/predict-json",
        json={
            "file_base64": _make_b64_volume([8, 8, 4]),
            "dimensions": [8, 8],
            "model_id": model_id,
        },
    )
    assert response.status_code == 400


def test_predict_json_rejects_unknown_model():
    client = TestClient(app)

    response = client.post(
        "/predict-json",
        json={
            "file_base64": _make_b64_volume([8, 8, 4]),
            "dimensions": [8, 8, 4],
            "model_id": "nonexistent",
        },
    )
    assert response.status_code == 404


# ---- jobs API tests ----

def test_submit_job_returns_job_id():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    def _fake(_mid, input_data):
        return (input_data > 0.5).astype(np.uint8)

    engine.run_inference = _fake

    response = client.post(
        "/jobs",
        json={
            "file_base64": _make_b64_volume([8, 8, 4]),
            "dimensions": [8, 8, 4],
            "model_id": model_id,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "running"
    assert body["model_id"] == model_id


def test_submit_job_rejects_unknown_model():
    client = TestClient(app)

    response = client.post(
        "/jobs",
        json={
            "file_base64": _make_b64_volume([8, 8, 4]),
            "dimensions": [8, 8, 4],
            "model_id": "nonexistent",
        },
    )
    assert response.status_code == 404


def test_get_job_completed_returns_result():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    def _fake(_mid, input_data):
        return (input_data > 0.5).astype(np.uint8)

    engine.run_inference = _fake

    response = client.post(
        "/jobs",
        json={
            "file_base64": _make_b64_volume([8, 8, 4]),
            "dimensions": [8, 8, 4],
            "model_id": model_id,
        },
    )
    job_id = response.json()["job_id"]

    # Wait for background task to finish
    for _ in range(50):
        time.sleep(0.1)
        r = client.get(f"/jobs/{job_id}")
        if r.json()["status"] in ("completed", "failed"):
            break

    body = r.json()
    assert body["status"] == "completed"
    assert body["result_nifti_base64"] is not None

    # Verify the base64 result is valid NIfTI
    nifti_bytes = base64.b64decode(body["result_nifti_base64"])
    with io.BytesIO(nifti_bytes) as bio:
        with gzip.GzipFile(fileobj=bio) as gz:
            nii_bytes = gz.read()
    img = nib.Nifti1Image.from_bytes(nii_bytes)
    assert tuple(img.shape) == (8, 8, 4)


def test_get_job_not_found():
    client = TestClient(app)
    response = client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_submit_job_rejects_bad_dimensions():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    response = client.post(
        "/jobs",
        json={
            "file_base64": _make_b64_volume([8, 8, 4]),
            "dimensions": [8, 8],
            "model_id": model_id,
        },
    )
    assert response.status_code == 400


# ---- health, version, model detail tests ----


def test_health_returns_ok():
    _seed_segmentation_model()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["loaded_models"] == 1


def test_version_returns_fields():
    client = TestClient(app)

    response = client.get("/version")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "neuroview-backend"
    assert "version" in body
    assert "model_pack_version" in body


def test_get_model_returns_detail():
    model_id = _seed_segmentation_model()
    client = TestClient(app)

    response = client.get(f"/models/{model_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == model_id
    assert body["name"] == "Test Seg Model"
    assert "description" in body


def test_get_model_not_found():
    client = TestClient(app)

    response = client.get("/models/nonexistent")
    assert response.status_code == 404
    assert response.json()["error"] == "Unknown model"
