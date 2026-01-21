import os
import json
import torch
import numpy as np
import tensorflow as tf
from typing import Dict, Any
from schemas import ModelMetadata, ModelEngine, ModelMethod, FinalLayer

class InferenceEngine:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        self.metadata_store: Dict[str, ModelMetadata] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Engine initialized on {self.device.upper()}")

    def load_all_models(self):
        """Scans the models directory and loads compatible models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"Created {self.models_dir} directory.")
            return

        print(f"🔍 Scanning '{self.models_dir}' for models...")
        
        for folder_name in os.listdir(self.models_dir):
            folder_path = os.path.join(self.models_dir, folder_name)
            config_path = os.path.join(folder_path, "model.json")

            if os.path.isdir(folder_path) and os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    # Validate metadata using Pydantic
                    meta = ModelMetadata(id=folder_name, **config_dict)
                    checkpoint_path = os.path.join(folder_path, meta.checkpoint_name)

                    if not os.path.exists(checkpoint_path):
                        print(f"❌ Checkpoint missing for {folder_name}: {checkpoint_path}")
                        continue

                    # Load actual model based on Engine
                    if meta.engine == ModelEngine.KERAS:
                        self.loaded_models[folder_name] = self._load_keras(checkpoint_path)
                    elif meta.engine == ModelEngine.TORCH:
                        self.loaded_models[folder_name] = self._load_torch(checkpoint_path)
                    
                    self.metadata_store[folder_name] = meta
                    print(f"✅ Loaded [{meta.engine.value.upper()}] {folder_name} ({meta.method.value})")

                except Exception as e:
                    print(f"❌ Error loading {folder_name}: {e}")

    def _load_keras(self, path):
        return tf.keras.models.load_model(path)

    def _load_torch(self, path):
        # We load TorchScript models for safe, code-free deployment
        model = torch.jit.load(path, map_location=self.device)
        model.eval()
        return model

    def run_inference(self, model_id: str, input_data: np.ndarray):
        """Standardizes input/output for both engines."""
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not found.")

        model = self.loaded_models[model_id]
        meta = self.metadata_store[model_id]
        
        # 1. PRE-PROCESS
        # Ensure standard (Batch, Channel, D, H, W) or (Batch, D, H, W, Channel)
        # Here we assume input is a raw 3D numpy array (D, H, W) -> add batch dim
        input_expanded = np.expand_dims(input_data, axis=0) 

        # 2. INFERENCE
        if meta.engine == ModelEngine.KERAS:
            # Keras expects (Batch, D, H, W, Channels) usually
            if len(input_expanded.shape) == 4: 
                input_expanded = np.expand_dims(input_expanded, axis=-1) # Add channel last
            
            raw_output = model.predict(input_expanded)
            
        elif meta.engine == ModelEngine.TORCH:
            # Torch expects (Batch, Channels, D, H, W)
            input_tensor = torch.tensor(input_expanded, dtype=torch.float32).unsqueeze(1).to(self.device)
            with torch.no_grad():
                raw_output = model(input_tensor)
                # Apply post-activation in Torch if model outputs logits
                if meta.final_layer == FinalLayer.SOFTMAX:
                    raw_output = torch.softmax(raw_output, dim=1)
                elif meta.final_layer == FinalLayer.SIGMOID:
                    raw_output = torch.sigmoid(raw_output)
                
                raw_output = raw_output.cpu().numpy()

        # 3. POST-PROCESS (Workflow Split)
        if meta.method == ModelMethod.SEGMENTATION:
            return self._process_segmentation(raw_output, meta)
        elif meta.method == ModelMethod.CLASSIFICATION:
            return self._process_classification(raw_output, meta)

    def _process_segmentation(self, output, meta: ModelMetadata):
        # Output likely shape: (1, Classes, D, H, W) or (1, D, H, W, Classes)
        
        # If binary, threshold at 0.5
        if meta.binary:
            mask = (output > 0.5).astype(np.uint8)
        else:
            # Multi-class: argmax along channel dimension
            # Heuristic to find channel dim: usually the smallest dim not 1, or explicit config
            # For now, assuming channel is axis 1 (Torch) or axis -1 (Keras)
            axis = 1 if meta.engine == ModelEngine.TORCH else -1
            mask = np.argmax(output, axis=axis).astype(np.uint8)
        
        return np.squeeze(mask) # Remove batch dim

    def _process_classification(self, output, meta: ModelMetadata):
        # Simple probability return
        return np.squeeze(output).tolist()