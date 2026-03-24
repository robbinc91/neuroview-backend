import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Settings:
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
    max_voxels: int = int(os.getenv("MAX_VOXELS", str(512 * 512 * 300)))
    inference_timeout_seconds: int = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "30"))
    allowed_origins_raw: str = os.getenv("ALLOWED_ORIGINS", "*")
    backend_version: str = os.getenv("BACKEND_VERSION", "1.0.0")
    model_pack_version: str = os.getenv("MODEL_PACK_VERSION", "dev")

    @property
    def allowed_origins(self) -> List[str]:
        raw = self.allowed_origins_raw.strip()
        if not raw:
            return ["*"]
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


settings = Settings()
