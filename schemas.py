from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator

class ModelEngine(str, Enum):
    KERAS = "keras"
    TORCH = "torch"

class ModelMethod(str, Enum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"

class FinalLayer(str, Enum):
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    NONE = "none"

class ModelMetadata(BaseModel):
    id: str
    checkpoint_name: str
    method: ModelMethod
    binary: Optional[bool] = False
    final_layer: Optional[FinalLayer] = FinalLayer.NONE
    engine: ModelEngine
    
    # Keras: Custom Objects mapping
    custom_objects: Optional[Dict[str, str]] = None 
    
    # PyTorch: Raw Model Support
    python_file: Optional[str] = None       # Filename (e.g., "my_architecture.py")
    class_name: Optional[str] = None        # Class Name (e.g., "MyUNet")
    model_args: Optional[Dict[str, Any]] = None # Init args (e.g., {"input_channels": 1})

    # Frontend-facing display fields (optional in model.json; defaults are generated)
    name: Optional[str] = None
    description: Optional[str] = None
    goal: Optional[str] = None
    input_shape: Optional[Union[str, List[int]]] = None
    input_format: Optional[str] = None
    output_classes: Optional[List[str]] = None


class ModelListItem(BaseModel):
    id: Union[str, int]
    name: str
    description: str
    goal: Optional[str] = None
    input_shape: Optional[Union[str, List[int]]] = None
    input_format: Optional[str] = None
    output_classes: Optional[List[str]] = None

    # compatibility aliases accepted by frontend normalizer
    method_id: Union[str, int]
    method_name: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None


class VersionResponse(BaseModel):
    service: str = Field(default="neuroview-backend")
    version: str
    model_pack_version: str


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictJsonRequest(BaseModel):
    file_base64: str
    dimensions: List[int]
    model_id: str

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_three(cls, v: List[int]) -> List[int]:
        if len(v) != 3:
            raise ValueError("dimensions must be a 3-element array [x, y, z]")
        return v


class JobCreateRequest(BaseModel):
    """Same shape as PredictJsonRequest — used for POST /jobs."""
    file_base64: str
    dimensions: List[int]
    model_id: str

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_three(cls, v: List[int]) -> List[int]:
        if len(v) != 3:
            raise ValueError("dimensions must be a 3-element array [x, y, z]")
        return v


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    model_id: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_nifti_base64: Optional[str] = None
    error: Optional[str] = None