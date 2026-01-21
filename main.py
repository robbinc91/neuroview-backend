import io
import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from engine import InferenceEngine
from schemas import ModelMethod

# --- LIFECYCLE MANAGEMENT ---
engine = InferenceEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    engine.load_all_models()
    yield
    # Shutdown logic if needed

app = FastAPI(lifespan=lifespan)

@app.get("/models")
def list_models():
    """Returns list of available loaded models and their metadata."""
    return engine.metadata_store

@app.post("/predict/{model_id}")
async def predict(model_id: str, file: UploadFile = File(...)):
    """Generic endpoint for both segmentation and classification."""
    
    if model_id not in engine.metadata_store:
        raise HTTPException(status_code=404, detail="Model not found")
    
    meta = engine.metadata_store[model_id]

    # 1. Read NIfTI
    try:
        content = await file.read()
        with io.BytesIO(content) as buffer:
            # Nibabel requires a file-like object with a name or a generic wrapper
            # We use a temp file strategy for stability with Nibabel
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=True) as tmp:
                tmp.write(content)
                tmp.flush()
                nii = nib.load(tmp.name)
                data = nii.get_fdata()
                affine = nii.affine
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid NIfTI file: {str(e)}")

    # 2. Run Inference
    try:
        result = engine.run_inference(model_id, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # 3. Return Result based on Method
    if meta.method == ModelMethod.SEGMENTATION:
        # Create NIfTI from result mask
        out_img = nib.Nifti1Image(result, affine)
        out_buffer = io.BytesIO()
        nib.save(out_img, nib.FileHolder(fileobj=out_buffer))
        out_buffer.seek(0)
        
        return StreamingResponse(
            out_buffer, 
            media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename={model_id}_result.nii.gz"}
        )
    
    elif meta.method == ModelMethod.CLASSIFICATION:
        return JSONResponse(content={"model": model_id, "prediction": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)