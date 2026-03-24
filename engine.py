import os
import json
import torch
import numpy as np
import importlib.util
import sys
from typing import Dict, Any, Optional
from schemas import ModelMetadata, ModelEngine, ModelMethod, FinalLayer

try:
    import tensorflow as tf
except Exception:
    tf = None

class InferenceEngine:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        self.metadata_store: Dict[str, ModelMetadata] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Engine initialized on {self.device.upper()}")

    def load_all_models(self):
        # ... (Same directory scanning logic as before) ...
        # Copy the scanning logic from the previous response
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            return

        print(f"🔍 Scanning '{self.models_dir}' for models...")
        
        for folder_name in os.listdir(self.models_dir):
            folder_path = os.path.join(self.models_dir, folder_name)
            config_path = os.path.join(folder_path, "model.json")

            if os.path.isdir(folder_path) and os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    meta = ModelMetadata(id=folder_name, **config_dict)
                    checkpoint_path = os.path.join(folder_path, meta.checkpoint_name)

                    if not os.path.exists(checkpoint_path):
                        print(f"❌ Checkpoint missing for {folder_name}")
                        continue

                    if meta.engine == ModelEngine.KERAS:
                        self.loaded_models[folder_name] = self._load_keras(checkpoint_path, meta, folder_path)
                    elif meta.engine == ModelEngine.TORCH:
                        self.loaded_models[folder_name] = self._load_torch(checkpoint_path, meta, folder_path)
                    
                    self.metadata_store[folder_name] = meta
                    print(f"✅ Loaded [{meta.engine.value.upper()}] {folder_name}")

                except Exception as e:
                    print(f"❌ Error loading {folder_name}: {e}")
                    import traceback
                    traceback.print_exc()

    def _dynamic_import(self, folder_path: str, filename: str, object_name: str):
        """Helper to load a class/function from a file dynamically."""
        fname = filename if filename.endswith(".py") else f"{filename}.py"
        file_path = os.path.join(folder_path, fname)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Python source file not found: {file_path}")
            
        module_name = filename.replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module 
        spec.loader.exec_module(module)
        
        if not hasattr(module, object_name):
            raise AttributeError(f"Object '{object_name}' not found in {file_path}")
            
        return getattr(module, object_name)

    def _load_keras(self, checkpoint_path: str, meta: ModelMetadata, folder_path: str):
        """
        Loads Keras model via:
        1. Python Class instantiation + load_weights (for Subclassed models)
        2. Standard load_model (for SavedModel/H5)
        """
        
        if tf is None:
            raise RuntimeError("TensorFlow is not installed but a Keras model is configured")

        # Option A: Python Class Definition (Subclassed Model)
        if meta.python_file and meta.class_name:
            print(f"   ↳ Instantiating Keras Subclass {meta.class_name} from {meta.python_file}...")
            
            # 1. Dynamically import the class
            ModelClass = self._dynamic_import(folder_path, meta.python_file, meta.class_name)
            
            # 2. Instantiate with args
            kwargs = meta.model_args if meta.model_args else {}
            model = ModelClass(**kwargs)
            
            # 3. Build the model (Optional but recommended for shape inference)
            # We construct a dummy input to build the graph if input_shape is known, 
            # otherwise load_weights usually handles it if the topology matches.
            try:
                # Attempt to load weights directly
                model.load_weights(checkpoint_path)
            except ValueError:
                # Sometimes Keras requires 'build' called first for subclassed models
                print("      ⚠️ Weights load failed, attempting to build model with dummy input...")
                # Assuming 3D input based on context, 1 channel. Adjust based on your standard.
                model.build(input_shape=(None, 64, 64, 64, 1)) 
                model.load_weights(checkpoint_path)
                
            return model

        # Option B: Standard Load (H5 or SavedModel)
        else:
            custom_objects_dict = {}
            if meta.custom_objects:
                for keras_name, python_name in meta.custom_objects.items():
                    try:
                        # Reusing dynamic import for consistency
                        custom_obj = self._dynamic_import(folder_path, f"{python_name}.py", python_name)
                        custom_objects_dict[keras_name] = custom_obj
                    except Exception as e:
                        print(f"   ⚠️ Failed to load custom object {python_name}: {e}")

            return tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects_dict)

    def _load_torch(self, checkpoint_path: str, meta: ModelMetadata, folder_path: str):
        # (Same PyTorch logic as previous response)
        if meta.python_file and meta.class_name:
            print(f"   ↳ Instantiating {meta.class_name} from {meta.python_file}...")
            ModelClass = self._dynamic_import(folder_path, meta.python_file, meta.class_name)
            kwargs = meta.model_args if meta.model_args else {}
            model = ModelClass(**kwargs)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        else:
            model = torch.jit.load(checkpoint_path, map_location=self.device)
            model.eval()
            return model

    def run_inference(self, model_id: str, input_data: np.ndarray):
        # (Same inference logic as previous response)
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not found.")

        model = self.loaded_models[model_id]
        meta = self.metadata_store[model_id]
        
        input_expanded = np.expand_dims(input_data, axis=0) 

        if meta.engine == ModelEngine.KERAS:
            # Add Channel dim if missing (Batch, D, H, W, 1)
            if len(input_expanded.shape) == 4: 
                input_expanded = np.expand_dims(input_expanded, axis=-1)
            raw_output = model.predict(input_expanded)
            
        elif meta.engine == ModelEngine.TORCH:
            input_tensor = torch.tensor(input_expanded, dtype=torch.float32).unsqueeze(1).to(self.device)
            with torch.no_grad():
                raw_output = model(input_tensor)
                if meta.final_layer == FinalLayer.SOFTMAX:
                    raw_output = torch.softmax(raw_output, dim=1)
                elif meta.final_layer == FinalLayer.SIGMOID:
                    raw_output = torch.sigmoid(raw_output)
                raw_output = raw_output.cpu().numpy()

        if meta.method == ModelMethod.SEGMENTATION:
            return self._process_segmentation(raw_output, meta)
        elif meta.method == ModelMethod.CLASSIFICATION:
            return self._process_classification(raw_output, meta)

    def _process_segmentation(self, output, meta: ModelMetadata):
        if meta.binary:
            mask = (output > 0.5).astype(np.uint8)
        else:
            axis = 1 if meta.engine == ModelEngine.TORCH else -1
            mask = np.argmax(output, axis=axis).astype(np.uint8)
        return np.squeeze(mask)

    def _process_classification(self, output, meta: ModelMetadata):
        return np.squeeze(output).tolist()