import os
from datetime import datetime, timezone, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from models import (
    Media,
    MediaUsage,
    User,
    Event,
    EventParticipant,
    FaceCluster,
    FaceEmbedding,
    ContentOwnerType,
    MediaUsageType,
)
from models.downloads import DownloadJob, DownloadJobStatus
from db import get_session
from utils.users import get_current_user

router = APIRouter(prefix="/downloads", tags=["downloads"])

DOWNLOAD_EXPIRY_HOURS = 1
MAX_FILES_PER_DOWNLOAD = 500


class BulkDownloadRequest(BaseModel):
    media_ids: List[int]


class ClusterDownloadRequest(BaseModel):
    cluster_id: int


class DownloadJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class DownloadStatusResponse(BaseModel):
    job_id: str
    status: str
    download_url: str | None
    error: str | None
    expires_at: datetime | None
    total_files: int | None
    created_at: datetime


def validate_media_access(session: Session, media_id: int, current_user: User) -> Media:
    """
    Validate that media exists, is gallery media, and user has access.
    Returns the Media object if valid, raises HTTPException otherwise.
    """
    media = session.get(Media, media_id)
    if not media:
        raise HTTPException(404, f"Media {media_id} not found")

    # Get usage to verify it's gallery media
    usage = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media_id)
    ).first()

    if not usage or usage.usage_type != MediaUsageType.GALLERY:
        raise HTTPException(400, f"Media {media_id} is not gallery media")

    if usage.owner_type != ContentOwnerType.EVENT:
        raise HTTPException(400, f"Media {media_id} is not event media")

    # Verify user has access to the event
    event = session.get(Event, usage.owner_id)
    if not event:
        raise HTTPException(404, f"Event not found for media {media_id}")

    # Check if user is creator or approved participant
    if event.created_by_id != current_user.id:
        participant = session.exec(
            select(EventParticipant).where(
                EventParticipant.event_id == event.id,
                EventParticipant.user_id == current_user.id,
                EventParticipant.status == "approved",
            )
        ).first()

        if not participant:
            raise HTTPException(
                403, f"You don't have access to event for media {media_id}"
            )

    return media


@router.post("/bulk", response_model=DownloadJobResponse)
async def create_bulk_download(
    request: BulkDownloadRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Create a bulk download job for event gallery media.
    Validates access and creates job, then queues background task.
    Returns a job ID to poll for status and download URL.
    """
    if not request.media_ids:
        raise HTTPException(400, "No media IDs provided")

    if len(request.media_ids) > MAX_FILES_PER_DOWNLOAD:
        raise HTTPException(400, f"Maximum {MAX_FILES_PER_DOWNLOAD} files per download")

    # Validate all media and user access
    for media_id in request.media_ids:
        validate_media_access(session, media_id, current_user)

    # Create download job with media_ids stored
    job = DownloadJob(
        user_id=current_user.id,
        status=DownloadJobStatus.PENDING,
        media_ids=request.media_ids,
        total_files=len(request.media_ids),
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=DOWNLOAD_EXPIRY_HOURS),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Queue Celery task with just the job_id
    from tasks.downloads import generate_bulk_download

    generate_bulk_download.delay(job.id)

    return DownloadJobResponse(
        job_id=job.id,
        status=job.status,
        message="Download job created. Poll /downloads/status/{job_id} for progress.",
    )


@router.post("/cluster", response_model=DownloadJobResponse)
async def create_cluster_download(
    request: ClusterDownloadRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Create a download job for all original images in a face cluster.
    User must have access to the event containing the cluster.
    """
    cluster = session.get(FaceCluster, request.cluster_id)
    if not cluster:
        raise HTTPException(404, "Cluster not found")

    # Verify user has access to the event
    if not cluster.event_id:
        raise HTTPException(400, "Cluster is not associated with an event")

    event = session.get(Event, cluster.event_id)
    if not event:
        raise HTTPException(404, "Event not found for cluster")

    # Check if user is creator or approved participant
    if event.created_by_id != current_user.id:
        participant = session.exec(
            select(EventParticipant).where(
                EventParticipant.event_id == event.id,
                EventParticipant.user_id == current_user.id,
                EventParticipant.status == "approved",
            )
        ).first()

        if not participant:
            raise HTTPException(403, "You don't have access to this event")

    # Get all unique media IDs from face embeddings in this cluster
    face_embeddings = session.exec(
        select(FaceEmbedding).where(FaceEmbedding.cluster_id == cluster.id)
    ).all()

    if not face_embeddings:
        raise HTTPException(404, "No faces found in this cluster")

    media_ids = list(set([fe.media_id for fe in face_embeddings]))

    if len(media_ids) > MAX_FILES_PER_DOWNLOAD:
        raise HTTPException(
            400,
            f"Cluster contains {len(media_ids)} images. Maximum {MAX_FILES_PER_DOWNLOAD} files per download",
        )

    # Create download job
    job = DownloadJob(
        user_id=current_user.id,
        status=DownloadJobStatus.PENDING,
        media_ids=media_ids,
        total_files=len(media_ids),
        cluster_id=cluster.id,
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=DOWNLOAD_EXPIRY_HOURS),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Queue Celery task
    from tasks.downloads import generate_bulk_download

    generate_bulk_download.delay(job.id)

    return DownloadJobResponse(
        job_id=job.id,
        status=job.status,
        message="Download job created. Poll /downloads/status/{job_id} for progress.",
    )


@router.get("/zip/{job_id}")
async def download_zip(
    job_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Download the generated ZIP file for a completed download job.
    File is deleted immediately after download.
    """
    job = session.get(DownloadJob, job_id)
    if not job:
        raise HTTPException(404, "Download job not found")

    # Verify ownership
    if job.user_id != current_user.id:
        raise HTTPException(403, "You don't have access to this download")

    # Check status
    if job.status != DownloadJobStatus.COMPLETED:
        raise HTTPException(400, f"Download is not ready. Status: {job.status}")

    # Check expiry
    if job.expires_at:
        now = datetime.now(timezone.utc)
        expires_at = (
            job.expires_at.replace(tzinfo=timezone.utc)
            if job.expires_at.tzinfo is None
            else job.expires_at
        )
        if expires_at < now:
            raise HTTPException(410, "Download link has expired")

    # Check if file exists
    if not job.file_path or not os.path.exists(job.file_path):
        raise HTTPException(404, "Download file not found")

    file_path = job.file_path

    # Determine filename based on cluster or bulk download
    if job.cluster_id:
        filename = f"cluster_{job.cluster_id}_photos.zip"
    else:
        filename = f"photos_{job.id[:8]}.zip"

    # Mark as deleted before serving
    job.status = DownloadJobStatus.DELETED
    job.file_path = None
    session.add(job)
    session.commit()

    # Return file with background cleanup
    from fastapi import BackgroundTasks

    background = BackgroundTasks()
    background.add_task(cleanup_file, file_path)

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/zip",
        },
        background=background,
    )


def cleanup_file(file_path: str):
    """Delete the file after it's been served."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up download file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")
