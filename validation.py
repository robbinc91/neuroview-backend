import json
from typing import Iterable, Tuple

import numpy as np


def parse_dimensions(dimensions_raw: str) -> Tuple[int, int, int]:
    try:
        parsed = json.loads(dimensions_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Expected dimensions as JSON array [x,y,z]") from exc

    if not isinstance(parsed, list) or len(parsed) != 3:
        raise ValueError("Expected dimensions as JSON array [x,y,z]")
    if not all(isinstance(v, int) and v > 0 for v in parsed):
        raise ValueError("Dimensions must be positive integers [x,y,z]")
    return int(parsed[0]), int(parsed[1]), int(parsed[2])


def validate_model_id(model_id: str, allowed_ids: Iterable[str]) -> None:
    if model_id not in set(allowed_ids):
        raise KeyError("Unknown model_id")


def _try_parse_with_dtype(file_bytes: bytes, voxel_count: int, dtype: np.dtype):
    dtype = np.dtype(dtype)
    expected = voxel_count * dtype.itemsize
    if len(file_bytes) != expected:
        return None
    arr = np.frombuffer(file_bytes, dtype=dtype, count=voxel_count)
    return arr


def parse_raw_volume(file_bytes: bytes, dims: Tuple[int, int, int], max_voxels: int) -> np.ndarray:
    voxel_count = dims[0] * dims[1] * dims[2]
    if voxel_count > max_voxels:
        raise ValueError("Input volume exceeds maximum allowed voxel count")

    # Conversion policy: float32 preferred, then signed/unsigned 16-bit
    array = None
    for dtype in (np.float32, np.int16, np.uint16):
        parsed = _try_parse_with_dtype(file_bytes, voxel_count, dtype)
        if parsed is not None:
            array = parsed
            break
    if array is None:
        raise ValueError("Invalid payload size for supported dtypes (float32/int16/uint16)")

    volume = array.astype(np.float32, copy=False).reshape(dims)
    return volume
