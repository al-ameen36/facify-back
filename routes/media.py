import os
import re
from datetime import datetime, timezone, timedelta
from fastapi import (
    APIRouter,
    Depends,
    Query,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
)
from sqlmodel import Session, delete, func, select
from db import get_session
from models import (
    User,
    UserRead,
    Media,
    MediaRead,
    MediaUsage,
    Event,
    EventParticipant,
    ContentOwnerType,
    MediaUsageType,
    PaginatedResponse,
    Pagination,
    SingleItemResponse,
)
from typing import Optional, Union
from dotenv import load_dotenv
from models.face import FaceMatch
from tasks.face import embed_media
from utils.users import get_current_user
from utils.media import upload_file, save_file_to_db, delete_media_and_file


load_dotenv()

FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME")
FACE_MODEL_BACKEND = os.environ.get("FACE_MODEL_BACKEND")
FACE_API_URL = os.environ.get("FACE_API_URL")

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}

# Rate limiting and storage limits
MAX_UPLOADS_PER_HOUR = 20
MAX_UPLOADS_PER_DAY = 100
MAX_USER_STORAGE_MB = 500  # 500MB per user
MAX_EVENT_MEDIA_COUNT = 200  # 200 media files per event

router = APIRouter(prefix="/uploads", tags=["media"])


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


@router.post(
    "", response_model=Union[SingleItemResponse[Media], SingleItemResponse[UserRead]]
)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    owner_id: int = Form(...),
    owner_type: str = Form(...),
    usage_type: str = Form(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Uploads a file to ImageKit for the user.
    - Saves Media + MediaUsage.
    - If image -> also generate and save embeddings (multiple faces supported).
    """
    # Rate Limiting: Check upload frequency limits
    check_rate_limits(session, current_user.id)

    # Content Validation: Sanitize filename
    if file.filename:
        file.filename = sanitize_filename(file.filename)

    # Get file size for storage limit checks
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    # Storage Limits: Check user storage quota
    check_storage_limits(session, current_user.id, file_size)

    # --- validate owner ---
    if owner_type == "user":
        owner = session.get(User, owner_id)
        # Users can only upload media for themselves
        if owner_id != current_user.id:
            raise HTTPException(403, "You can only upload media for yourself")
    elif owner_type == "event":
        owner = session.get(Event, owner_id)
        if not owner:
            raise HTTPException(404, "Event not found")

        # Event State Guards: Prevent uploads to ended events
        now = datetime.now(timezone.utc)
        if owner.end_time:
            end_time_aware = (
                owner.end_time.replace(tzinfo=timezone.utc)
                if owner.end_time.tzinfo is None
                else owner.end_time
            )
            if end_time_aware < now:
                raise HTTPException(
                    403, "Cannot upload media to an event that has already ended"
                )

        # For events, check authorization based on usage type
        if usage_type == MediaUsageType.COVER_PHOTO:
            # Only event creator can upload cover photos
            if owner.created_by_id != current_user.id:
                raise HTTPException(403, "Only event creators can upload cover photos")
        elif usage_type == MediaUsageType.GALLERY:
            # Check if event allows contributions
            if not owner.allow_contributions:
                raise HTTPException(
                    403, "This event does not allow media contributions"
                )

            # Storage Limits: Check event media count limits
            check_event_media_limits(session, owner_id)

            # Event creator or approved participants can upload to gallery
            if owner.created_by_id != current_user.id:
                participant_check = session.exec(
                    select(EventParticipant).where(
                        EventParticipant.event_id == owner_id,
                        EventParticipant.user_id == current_user.id,
                    )
                ).first()

                if not participant_check:
                    raise HTTPException(403, "You are not a participant of this event")
                elif participant_check.status == "pending":
                    raise HTTPException(
                        403, "Your participation request is still pending approval"
                    )
                elif participant_check.status == "rejected":
                    raise HTTPException(403, "Your participation request was rejected")
                elif participant_check.status != "approved":
                    raise HTTPException(403, "Access denied")
    else:
        raise HTTPException(400, "Invalid owner_type")

    if not owner:
        raise HTTPException(404, f"{owner_type.capitalize()} not found")

    supported_usage_types = [item.value for item in MediaUsageType]
    if usage_type not in supported_usage_types:
        raise HTTPException(400, f"Unsupported usage_type: {usage_type}")

    # MIME Type Validation: Enforce allowed file types
    if file.content_type:
        if file.content_type.startswith("image/"):
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    400,
                    f"Unsupported image type: {file.content_type}. Allowed types: {', '.join(ALLOWED_IMAGE_TYPES)}",
                )
        elif file.content_type.startswith("video/"):
            if file.content_type not in ALLOWED_VIDEO_TYPES:
                raise HTTPException(
                    400,
                    f"Unsupported video type: {file.content_type}. Allowed types: {', '.join(ALLOWED_VIDEO_TYPES)}",
                )
        else:
            raise HTTPException(
                400,
                f"Unsupported file type: {file.content_type}. Only images and videos are allowed.",
            )
    else:
        raise HTTPException(400, "File content type could not be determined")

    try:
        # if cover_photo or profile_picture: delete existing one
        if usage_type in [MediaUsageType.COVER_PHOTO, MediaUsageType.PROFILE_PICTURE]:
            old_usage = session.exec(
                select(MediaUsage).where(
                    MediaUsage.owner_id == owner_id,
                    MediaUsage.owner_type == owner_type,
                    MediaUsage.usage_type == usage_type,
                )
            ).first()

            if old_usage:
                delete_media_and_file(session, old_usage.media)
                session.commit()

        # Upload file first
        uploaded_media = upload_file(
            file=file,
            user=current_user,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
        )

        # Save to database without embeddings
        saved_media = save_file_to_db(
            session=session,
            media=uploaded_media,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
            embeddings=None,  # Will be generated in background
            user_id=current_user.id,
        )

        # Queue embedding generation in background (skip for cover photos)
        if usage_type != MediaUsageType.COVER_PHOTO:
            embed_media.delay(saved_media.id, uploaded_media.external_url)

        # If uploading profile picture, return updated user data
        if usage_type == MediaUsageType.PROFILE_PICTURE and owner_type == "user":
            session.refresh(current_user)  # Refresh to get latest data
            user_data = current_user.to_user_read(session)
            return SingleItemResponse(
                data=user_data, message="Profile picture updated successfully"
            )

        return SingleItemResponse(
            data=saved_media, message="Successfully uploaded media"
        )

    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Unexpected error: {str(e)}")


@router.get("/me", response_model=PaginatedResponse[MediaRead])
def get_user_uploads(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    # Count total gallery uploads
    total = session.exec(
        select(func.count(Media.id))
        .join(MediaUsage, Media.id == MediaUsage.media_id)
        .where(
            Media.uploaded_by_id == current_user.id, MediaUsage.usage_type == "gallery"
        )
    ).one()

    offset = (page - 1) * per_page

    # Get gallery uploads with pagination
    user_uploads = session.exec(
        select(Media)
        .join(MediaUsage, Media.id == MediaUsage.media_id)
        .where(
            Media.uploaded_by_id == current_user.id, MediaUsage.usage_type == "gallery"
        )
        .order_by(Media.created_at.desc())
        .offset(offset)
        .limit(per_page)
    ).all()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    # Convert to MediaRead objects
    media_reads = [media.to_media_read() for media in user_uploads]

    return PaginatedResponse[MediaRead](
        message="User uploads retrieved successfully",
        data=media_reads,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
        ),
    )


@router.get("/event/{event_id}", response_model=PaginatedResponse[MediaRead])
async def get_event_media(
    event_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    media_type: Optional[str] = Query(
        None, description="Filter by media type: image or video"
    ),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    # Check if event exists
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Event Privacy Guards: Restrict gallery access based on event privacy
    if event.privacy == "private":
        # Private events: only creator or approved participants can access gallery
        if event.created_by_id != current_user.id:
            participant_check = session.exec(
                select(EventParticipant).where(
                    EventParticipant.event_id == event_id,
                    EventParticipant.user_id == current_user.id,
                )
            ).first()

            if not participant_check:
                raise HTTPException(
                    status_code=403,
                    detail="You are not a participant of this private event",
                )
            elif participant_check.status == "pending":
                raise HTTPException(
                    status_code=403,
                    detail="Your participation request is still pending approval",
                )
            elif participant_check.status == "rejected":
                raise HTTPException(
                    status_code=403, detail="Your participation request was rejected"
                )
            elif participant_check.status != "approved":
                raise HTTPException(status_code=403, detail="Access denied")
    elif event.privacy == "public":
        # Public events: anyone can view the gallery (no additional checks needed)
        pass
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

    # Convert to MediaRead objects
    media_reads = [media.to_media_read() for media in event_media]

    return PaginatedResponse[MediaRead](
        message="Event media retrieved successfully",
        data=media_reads,
        pagination=Pagination(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        ),
    )


@router.delete("/{media_id}", response_model=dict)
async def delete_media(
    media_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Delete a media file and all associated data.
    Users can only delete media they uploaded or media from events they created.
    """
    # Get the media record
    media = session.get(Media, media_id)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    # Get media usage to check ownership/permissions
    media_usage = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media_id)
    ).first()

    if not media_usage:
        raise HTTPException(status_code=404, detail="Media usage not found")

    # Authorization checks
    can_delete = False

    # Check if user uploaded the media
    if media.uploaded_by_id == current_user.id:
        can_delete = True

    # Check if user owns the content (event creator for event media)
    elif media_usage.owner_type == ContentOwnerType.EVENT:
        event = session.get(Event, media_usage.owner_id)
        if event and event.created_by_id == current_user.id:
            can_delete = True

    # Check if user owns their own profile picture
    elif (
        media_usage.owner_type == ContentOwnerType.USER
        and media_usage.owner_id == current_user.id
    ):
        can_delete = True

    if not can_delete:
        raise HTTPException(
            status_code=403, detail="You don't have permission to delete this media"
        )

    try:
        # Delete dependent facematch entries
        session.exec(delete(FaceMatch).where(FaceMatch.media_id == media.id))
        session.commit()

        # Delete media and associated files
        delete_media_and_file(session, media)
        session.commit()

        return {"message": "Media deleted successfully", "media_id": media_id}

    except PermissionError as e:
        session.rollback()
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete media: {str(e)}")
