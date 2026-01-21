from enum import Enum
from typing import Optional
from pydantic import BaseModel

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
    id: str  # The folder name acts as ID
    checkpoint_name: str
    method: ModelMethod
    binary: Optional[bool] = False
    final_layer: Optional[FinalLayer] = FinalLayer.NONE
    engine: ModelEngine