from datetime import datetime, timezone
from typing import List, Optional
from sqlmodel import JSON, Column, Field, SQLModel
import uuid


class BulkDownloadRequest(SQLModel):
    media_ids: List[int]


class ClusterDownloadRequest(SQLModel):
    cluster_id: int


class DownloadJobResponse(SQLModel):
    job_id: str
    status: str
    message: str


class DownloadStatusResponse(SQLModel):
    job_id: str
    status: str
    download_url: str | None
    error: str | None
    expires_at: datetime | None
    total_files: int | None
    created_at: datetime


class DownloadJobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DELETED = "deleted"


class DownloadJob(SQLModel, table=True):
    __tablename__ = "download_jobs"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), primary_key=True, index=True
    )
    user_id: int = Field(foreign_key="user.id", index=True)
    status: str = Field(default=DownloadJobStatus.PENDING, index=True)
    total_files: int = Field(default=0)
    file_path: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    cluster_id: Optional[int] = Field(default=None, foreign_key="facecluster.id")
    media_ids: List[int] = Field(sa_column=Column(JSON), default=[])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(default=None, index=True)
