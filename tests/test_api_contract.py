import gzip
import io
import json

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
