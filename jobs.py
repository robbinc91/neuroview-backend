import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

from schemas import JobStatus


@dataclass
class JobRecord:
    job_id: str
    model_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_nifti_base64: Optional[str] = None
    error: Optional[str] = None


_job_store: dict[str, "JobRecord"] = {}
_lock = threading.Lock()


def create_job(model_id: str) -> JobRecord:
    job_id = str(uuid4())
    record = JobRecord(job_id=job_id, model_id=model_id)
    with _lock:
        _job_store[job_id] = record
    return record


def get_job(job_id: str) -> Optional[JobRecord]:
    with _lock:
        return _job_store.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        record = _job_store.get(job_id)
        if record is None:
            return
        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)


def cleanup_expired_jobs(max_age_seconds: float) -> int:
    now = time.time()
    with _lock:
        expired = [
            jid for jid, rec in _job_store.items()
            if now - rec.created_at > max_age_seconds
        ]
        for jid in expired:
            del _job_store[jid]
    return len(expired)


def store_count() -> int:
    with _lock:
        return len(_job_store)
