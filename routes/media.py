import os
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, func, select
from db import get_session
from models import User, Event, Media, SingleItemResponse
from models.core import ContentOwnerType, MediaUsageType, PaginatedResponse, Pagination
from models.media import MediaUsage
from utils.users import get_current_user
from typing import Optional
import mimetypes
from moviepy import VideoFileClip
from dotenv import load_dotenv


load_dotenv()

router = APIRouter(prefix="/uploads", tags=["media"])

MEDIA_DIR = os.environ.get("MEDIA_DIR")
os.makedirs(MEDIA_DIR, exist_ok=True)

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}


@router.post("", response_model=SingleItemResponse[Media])
async def upload_media(
    file: UploadFile = File(...),
    owner_id: int = Form(...),
    owner_type: str = Form(...),
    usage_type: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # Validate usage_type
    supported_usage_types = [item.value for item in MediaUsageType]
    if usage_type not in supported_usage_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported usage_type. Supported types: {supported_usage_types}",
        )

    # Validate owner_type
    if owner_type == "user":
        owner = session.get(User, owner_id)
        if not owner:
            raise HTTPException(status_code=404, detail="User not found")
    elif owner_type == "event":
        owner = session.get(Event, owner_id)
        if not owner:
            raise HTTPException(status_code=404, detail="Event not found")
    else:
        raise HTTPException(status_code=400, detail="Invalid owner_type")

    # Delete existing media for specific usage types that should be unique
    unique_usage_types = ["profile_picture", "cover_photo"]
    if usage_type in unique_usage_types:
        # Find existing media usage for this owner and usage type
        existing_usage = (
            session.query(MediaUsage)
            .filter(
                MediaUsage.owner_id == owner_id,
                MediaUsage.owner_type == owner_type,
                MediaUsage.usage_type == usage_type,
            )
            .first()
        )

        if existing_usage:
            # Get the associated media
            existing_media = session.get(Media, existing_usage.media_id)

            if existing_media:
                # Delete the physical file
                old_file_path = os.path.join(MEDIA_DIR, existing_media.filename)
                try:
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                except Exception as e:
                    # Log the error but continue - don't fail the upload if file deletion fails
                    print(f"Warning: Failed to delete old file {old_file_path}: {e}")

                # Delete the media usage record
                session.delete(existing_usage)

                # Delete the media record
                session.delete(existing_media)

                session.commit()

    # Save file to disk
    file_ext = os.path.splitext(file.filename)[1].lower()
    safe_filename = f"{owner_type}_{owner_id}_{os.urandom(8).hex()}{file_ext}"
    file_path = os.path.join(MEDIA_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Derive metadata
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = file.content_type or "application/octet-stream"

    file_size = os.path.getsize(file_path)
    duration = None

    # If video, extract duration
    if mime_type.startswith("video/"):
        try:
            with VideoFileClip(file_path) as clip:
                duration = clip.duration
        except Exception:
            duration = None

    # Create a URL
    file_url = f"/media/{safe_filename}"

    # Save Media
    media = Media(
        url=file_url,
        filename=safe_filename,
        original_filename=file.filename,
        file_size=file_size,
        mime_type=mime_type,
        duration=duration,
        uploaded_by_id=current_user.id,
    )
    session.add(media)
    session.commit()
    session.refresh(media)

    # Save MediaUsage
    usage = MediaUsage(
        content_type=owner_type,
        owner_id=owner_id,
        media_type=media.mime_type.split("/")[0],
        usage_type=usage_type,
        media_id=media.id,
        owner_type=owner_type,
    )
    session.add(usage)
    session.commit()
    session.refresh(media)

    return SingleItemResponse(data=media, message="Media uploaded successfully")


@router.get("/me", response_model=PaginatedResponse[Media])
def get_user_uploads(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    total = session.exec(select(func.count(Media.id))).one()
    offset = (page - 1) * per_page
    user_uploads = session.exec(
        select(Media)
        .filter(Media.uploaded_by_id == current_user.id)
        .order_by(Media.created_at.desc())
        .offset(offset)
        .limit(per_page)
    ).all()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[Media](
        message="User uploads retrieved successfully",
        data=user_uploads,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
        ),
    )


@router.get("/event/{event_id}", response_model=PaginatedResponse[Media])
async def get_event_media(
    event_id: int,
    session: Session = Depends(get_session),
    media_type: Optional[str] = Query(
        None, description="Filter by media type: image or video"
    ),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    # Base query: select Media via MediaUsage
    query = (
        select(Media)
        .join(MediaUsage, MediaUsage.media_id == Media.id)
        .where(
            MediaUsage.owner_id == event_id,
            MediaUsage.owner_type == ContentOwnerType.EVENT,
            MediaUsage.usage_type == MediaUsageType.GALLERY,
        )
    )

    # Apply filter if media_type is specified
    if media_type == "image":
        query = query.where(Media.mime_type.like("image/%"))
    elif media_type == "video":
        query = query.where(Media.mime_type.like("video/%"))

    # Count total
    total = session.exec(select(func.count()).select_from(query.subquery())).one()

    # Apply pagination
    offset = (page - 1) * per_page
    event_media = session.exec(
        query.order_by(Media.id.desc()).offset(offset).limit(per_page)
    ).all()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[Media](
        message="Event media retrieved successfully",
        data=event_media,
        pagination=Pagination(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        ),
    )
