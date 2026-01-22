from enum import Enum
from typing import Optional, Dict, Any
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