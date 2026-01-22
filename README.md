# 🧠 NeuroView AI Engine

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-teal) ![PyTorch](https://img.shields.io/badge/PyTorch-Supported-orange) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Supported-orange)

**NeuroView AI Engine** is a high-performance, framework-agnostic inference server designed for medical imaging analysis. It provides a unified REST API to run deep learning models built in **PyTorch** or **TensorFlow/Keras** on NIfTI MRI data.

This service is designed to be **frontend-agnostic**, serving segmentation masks and classification results to any connected client (Web, Desktop, or CLI).

---

## ✨ Core Features

* **Hybrid Engine:** Seamlessly loads and runs both **PyTorch** and **TensorFlow (Keras)** models in the same environment.
* **Flexible Loading Strategies:** Supports loading models via:
    * **Compiled/Saved Files:** TorchScript (`.pt`) or Keras SavedModel/H5.
    * **Source Code:** Raw Python class definitions + Weight files (State Dicts).
* **Dynamic Model Discovery:** Automatically scans the `/models` directory at startup to register available checkpoints. No code changes required to add new models.
* **Custom Object Support:** Dynamically imports and registers custom Keras layers/loss functions defined in external Python scripts.
* **Workflow Abstraction:** Automatically handles pre-processing and post-processing differences between frameworks, exposing a standardized API for:
    * **Segmentation:** Returns compressed NIfTI (`.nii.gz`) masks.
    * **Classification:** Returns JSON probability maps.
* **Hot-Pluggable Architecture:** Each model is self-contained in its own directory with its own configuration.

---

## 🛠️ Installation

### Prerequisites

* Python 3.9+
* CUDA Toolkit (optional, for GPU acceleration)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/robbinc91/neuroview-backend
    cd neuroview-backend
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server:**
    ```bash
    python main.py
    # OR directly via Uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```

---

## 📂 Model Management

To add a new model, you do not need to modify the server code. Simply create a new folder inside `models/` and add a `model.json` configuration file.

### Directory Structure

Each model must have its own folder containing the checkpoint, configuration, and any custom scripts.

```text
models/
├── brain_tumor_seg/           # Standard TorchScript Model
│   ├── model.json             # Configuration
│   └── checkpoint.pt          # PyTorch TorchScript file
│
├── lung_nodule_keras/         # Keras Model with Custom Objects
│   ├── model.json             # Configuration
│   ├── unet_v2.h5             # Keras H5 file
│   └── my_metrics.py          # Custom Python script
│
└── experimental_torch/        # PyTorch Model from Source Class
    ├── model.json             # Configuration
    ├── weights.pth            # State Dict
    └── architecture.py        # Python file defining the class

```

### Configuration (`model.json`)

The `model.json` file tells the engine how to load and run the model.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `checkpoint_name` | `string` | Yes | Filename of the model weight file (e.g., `model.pt`, `weights.h5`). |
| `engine` | `string` | Yes | Framework used: `"torch"` or `"keras"`. |
| `method` | `string` | Yes | Workflow type: `"segmentation"` or `"classification"`. |
| `binary` | `boolean` | No | (Segmentation only) `true` if output is binary (0/1), `false` for multi-class. |
| `final_layer` | `string` | No | Activation to apply if missing from model: `"sigmoid"`, `"softmax"`, or `"none"`. |
| `custom_objects` | `dict` | No | Map of Keras object names to Python filenames. |
| **Advanced** |  |  | **For loading models from Python Class Definitions:** |
| `python_file` | `string` | No | Filename containing the model class definition (e.g., `arch.py`). |
| `class_name` | `string` | No | The name of the class to instantiate (e.g., `UNet`). |
| `model_args` | `dict` | No | Arguments to pass to the class constructor (e.g., `{"in_channels": 1}`). |

---

## 🏗️ Advanced: Loading Models from Source

The engine supports loading models defined in raw Python code, which is useful when you have the weights (`state_dict`) but not a compiled TorchScript or Keras SavedModel.

### Scenario A: PyTorch Class + State Dict

If you have a `weights.pth` and a `model.py` file:

1. `models/my_model/model.json**`

```json
{
  "checkpoint_name": "weights.pth",
  "method": "segmentation",
  "engine": "torch",
  "python_file": "architecture.py",
  "class_name": "MyUNet",
  "model_args": {
    "in_channels": 1,
    "classes": 2
  }
}

```

2. `models/my_model/architecture.py**`

```python
import torch.nn as nn
class MyUNet(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        # ... definition

```

### Scenario B: Keras Subclassed Model + Weights

If you use Model Subclassing in Keras and only have weights:

1. `models/keras_sub/model.json**`

```json
{
  "checkpoint_name": "weights.h5",
  "method": "classification",
  "engine": "keras",
  "python_file": "resnet3d.py",
  "class_name": "ResNet3D",
  "model_args": { "depth": 50 }
}

```

---

## 🔌 API Reference

### 1. List Available Models

Retrieves the metadata for all loaded models.

* **Endpoint:** `GET /models`
* **Response:**
```json
{
  "brain_tumor_seg": {
    "id": "brain_tumor_seg",
    "checkpoint_name": "checkpoint.pt",
    "method": "segmentation",
    "engine": "torch",
    ...
  }
}

```



### 2. Run Inference

Uploads a NIfTI volume and returns the processed result.

* **Endpoint:** `POST /predict/{model_id}`
* **Parameters:**
* `model_id`: The folder name of the model to use.
* `file`: The NIfTI file (`.nii` or `.nii.gz`) uploaded as form-data.



#### Scenario A: Segmentation

Returns a streaming NIfTI file.

* **Response Header:** `Content-Type: application/gzip`
* **Body:** Binary `.nii.gz` file content.

#### Scenario B: Classification

Returns a JSON object with predictions.

* **Response Header:** `Content-Type: application/json`
* **Body:**
```json
{
  "model": "lung_nodule_classifier",
  "prediction": [0.05, 0.95] 
}

```



---

## 🧩 Custom Objects (Keras)

If your Keras model uses custom layers, metrics, or loss functions, you must define them in separate Python files inside the model's directory.

**Rules:**

1. The Python file must be named exactly as referenced in `model.json` (or map the key in `model.json` to the filename).
2. The function or class inside the Python file must match the filename.

**Example:**
If `model.json` contains `"dice_score": "my_metric"`, the engine looks for `models/{model_id}/my_metric.py` and imports `my_metric`.

```python
# my_metric.py
import tensorflow as tf

def my_metric(y_true, y_pred):
    return score

```

---

## ⚠️ Limitations & Notes

* **Input Dimensions:** The engine automatically adds a batch dimension `(1, D, H, W)` before inference and removes it after. Ensure your models expect 4D (or 5D for Keras channel-last) inputs.
* **Concurrency:** Inference is currently blocking (per request). For high-load production environments, consider integrating a task queue (Celery/Redis) behind this API.
* **Security:** Loading models from Python source files involves dynamic execution of code (`importlib`). Ensure you only add models from trusted sources to the `/models` directory.

---

## 📄 License

Distributed under the MIT License.
