from datetime import datetime, timedelta, timezone
import os
import re
import shutil
import tempfile
from typing import List
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from dotenv import load_dotenv
from sqlmodel import Session, func, select
from fastapi import HTTPException, UploadFile
from models import (
    User,
    Media,
    MediaUsage,
    FaceEmbedding,
    ContentOwnerType,
    MediaUsageType,
)
import mimetypes

load_dotenv()

# Temp
MAX_UPLOADS_PER_HOUR = 20
MAX_UPLOADS_PER_DAY = 100
MAX_USER_STORAGE_MB = 500  # 500MB per user
MAX_EVENT_MEDIA_COUNT = 200  # 200 media files per event

PRIVATE_KEY = os.environ.get("IK_PRIVATE_KEY")
PUBLIC_KEY = os.environ.get("IK_PUBLIC_KEY")
URL = os.environ.get("IK_URL")
FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME", "ArcFace")

imagekit = ImageKit(private_key=PRIVATE_KEY, public_key=PUBLIC_KEY, url_endpoint=URL)


def get_mime_type(file_path: str) -> str:
    """
    Returns the MIME type of a file based on its extension.
    Falls back to 'application/octet-stream' if unknown.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def upload_file(
    file: UploadFile,
    user: User,
    owner_id: int,
    owner_type: str,
    usage_type: str,
    use_unique_file_name: bool = True,
):
    # File size limit: 50MB for videos, 10MB for images
    MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

    # Get file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    # Check file size limits based on content type
    if file.content_type and file.content_type.startswith("video/"):
        if file_size > MAX_VIDEO_SIZE:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=413,
                detail=f"Video file too large. Maximum size is {MAX_VIDEO_SIZE // (1024*1024)}MB",
            )
    elif file.content_type and file.content_type.startswith("image/"):
        if file_size > MAX_IMAGE_SIZE:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=413,
                detail=f"Image file too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB",
            )
    else:
        # Default limit for unknown file types
        if file_size > MAX_IMAGE_SIZE:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_IMAGE_SIZE // (1024*1024)}MB",
            )

    # Always use temp file approach for consistency and memory efficiency
    return _upload_via_temp_file(
        file, user, owner_id, owner_type, usage_type, use_unique_file_name
    )


def _upload_via_temp_file(
    file, user, owner_id, owner_type, usage_type, use_unique_file_name
):
    """Upload large files via temp file (memory efficient)"""
    file_path = save_upload_file_to_temp(file)

    try:
        with open(file_path, "rb") as f:
            upload = imagekit.upload_file(
                file=f,
                file_name=file.filename or "uploaded_file",
                options=UploadFileRequestOptions(
                    use_unique_file_name=use_unique_file_name,
                    tags=[
                        owner_type,
                        f"owner-{owner_id}",
                        f"creator-{user.id}",
                        f"usage-{usage_type}",
                    ],
                ),
            )
    finally:
        # Clean up temp file
        try:
            os.remove(file_path)
        except OSError:
            pass

    return _process_upload_response(upload, file, user.id)


def _process_upload_response(upload, file, user_id):
    """Process ImageKit upload response into Media object"""
    raw = upload.response_metadata.raw

    media_data = {
        "external_id": raw.get("fileId"),
        "external_url": raw.get("url"),
        "filename": raw.get("name", "unknown"),
        "original_filename": file.filename,
        "file_size": raw.get("size"),
        "mime_type": get_mime_type(file.filename or ""),
        "duration": raw.get("duration", 0),
        "uploaded_by_id": user_id,
    }

    return Media(**media_data)


def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """
    Save a FastAPI UploadFile to a temporary file and return its path.
    """
    suffix = (
        upload_file.filename.split(".")[-1]
        if upload_file.filename and "." in upload_file.filename
        else ""
    )
    suffix = f".{suffix}" if suffix else ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, tmp)
        temp_file_path = tmp.name

    return temp_file_path


def save_file_to_db(
    session: Session,
    media: Media,
    owner_id: int,
    owner_type: str,
    usage_type: str,
    user_id: int = None,
    embeddings: List[List[float]] = None,
    approval_status: str = "approved",
    tags: str = "",
):
    media.tags = tags
    session.add(media)
    session.flush()

    usage = MediaUsage(
        owner_id=owner_id,
        owner_type=owner_type,
        usage_type=usage_type,
        media_type=media.mime_type,
        media_id=media.id,
        approval_status=approval_status,
        tags=tags,
    )
    session.add(usage)
    session.flush()

    existing_embedding = session.exec(
        select(FaceEmbedding).where(FaceEmbedding.media_id == media.id)
    ).first()
    if not existing_embedding:
        media_embedding = FaceEmbedding(
            media_id=media.id,
            model_name=FACE_MODEL_NAME,
            user_id=(user_id if MediaUsageType.PROFILE_PICTURE else None),
            embeddings=embeddings,
            status="completed" if embeddings else "pending",
        )
        session.add(media_embedding)
        session.commit()

    return media


def delete_media_and_file(session: Session, media: Media):
    """Delete a media row + usage + embeddings + ImageKit file."""
    if not media:
        return

    # Delete from ImageKit
    if media.external_id:
        try:
            imagekit.delete_file(file_id=media.external_id)
        except Exception:
            pass  # file may already be gone

    # Delete embeddings
    embeddings = session.exec(
        select(FaceEmbedding).where(FaceEmbedding.media_id == media.id)
    ).all()
    for embedding in embeddings:
        session.delete(embedding)

    # Delete usages
    usages = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media.id)
    ).all()
    for usage in usages:
        session.delete(usage)

    # Finally delete the media
    session.delete(media)
    session.flush()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent security issues"""
    if not filename:
        return "unnamed_file"

    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove control characters
    filename = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename)
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext

    return filename or "unnamed_file"


def check_rate_limits(session: Session, user_id: int) -> None:
    """Check if user has exceeded rate limits"""
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(days=1)

    # Check hourly limit
    hourly_uploads = session.exec(
        select(func.count(Media.id)).where(
            Media.uploaded_by_id == user_id, Media.created_at >= hour_ago
        )
    ).one()

    if hourly_uploads >= MAX_UPLOADS_PER_HOUR:
        raise HTTPException(
            429, f"Rate limit exceeded: Maximum {MAX_UPLOADS_PER_HOUR} uploads per hour"
        )

    # Check daily limit
    daily_uploads = session.exec(
        select(func.count(Media.id)).where(
            Media.uploaded_by_id == user_id, Media.created_at >= day_ago
        )
    ).one()

    if daily_uploads >= MAX_UPLOADS_PER_DAY:
        raise HTTPException(
            429, f"Rate limit exceeded: Maximum {MAX_UPLOADS_PER_DAY} uploads per day"
        )


def check_storage_limits(session: Session, user_id: int, new_file_size: int) -> None:
    """Check if user has exceeded storage limits"""
    # Get current user storage usage
    current_usage = session.exec(
        select(func.coalesce(func.sum(Media.file_size), 0)).where(
            Media.uploaded_by_id == user_id
        )
    ).one()

    max_storage_bytes = MAX_USER_STORAGE_MB * 1024 * 1024

    if current_usage + new_file_size > max_storage_bytes:
        current_mb = current_usage / (1024 * 1024)
        raise HTTPException(
            413,
            f"Storage limit exceeded: You have used {current_mb:.1f}MB of {MAX_USER_STORAGE_MB}MB",
        )


def check_event_media_limits(session: Session, event_id: int) -> None:
    """Check if event has exceeded media count limits"""
    media_count = session.exec(
        select(func.count(MediaUsage.id)).where(
            MediaUsage.owner_type == ContentOwnerType.EVENT,
            MediaUsage.owner_id == event_id,
            MediaUsage.usage_type == MediaUsageType.GALLERY,
        )
    ).one()

    if media_count >= MAX_EVENT_MEDIA_COUNT:
        raise HTTPException(
            413,
            f"Event media limit exceeded: Maximum {MAX_EVENT_MEDIA_COUNT} media files per event",
        )
