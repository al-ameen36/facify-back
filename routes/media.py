import os
from datetime import datetime, timezone
from fastapi import (
    APIRouter,
    Depends,
    Query,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from tasks.notifications import send_notification
from sqlmodel import Session, delete, func, select
from db import get_session
from models import (
    User,
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
from typing import Optional
from dotenv import load_dotenv
from tasks.face import embed_media
from utils.users import get_current_user
from utils.media import (
    check_event_media_limits,
    check_rate_limits,
    check_storage_limits,
    sanitize_filename,
    upload_file,
    save_file_to_db,
    delete_media_and_file,
)


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


@router.post(
    "",
    response_model=SingleItemResponse[Media],
)
async def upload_media(
    file: UploadFile = File(...),
    owner_id: int = Form(...),
    owner_type: str = Form(...),
    usage_type: str = Form(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Uploads a file (image or video) to ImageKit for a user or event.
    - Handles cover photos and event gallery uploads.
    - Enforces authorization and upload limits.
    - Triggers embeddings and notifications as needed.
    """
    # --- Basic validations ---
    check_rate_limits(session, current_user.id)

    if file.filename:
        file.filename = sanitize_filename(file.filename)

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    check_storage_limits(session, current_user.id, file_size)

    if owner_type == "user":
        raise HTTPException(400, "Direct user media uploads are no longer supported.")
    elif owner_type == "event":
        owner = session.get(Event, owner_id)
        if not owner:
            raise HTTPException(404, "Event not found")

        now = datetime.now(timezone.utc)
        if owner.end_time:
            end_time_aware = (
                owner.end_time.replace(tzinfo=timezone.utc)
                if owner.end_time.tzinfo is None
                else owner.end_time
            )
            if end_time_aware < now:
                raise HTTPException(403, "Cannot upload media to a past event")

        # Handle different usage types
        if usage_type == MediaUsageType.COVER_PHOTO:
            if owner.created_by_id != current_user.id:
                raise HTTPException(403, "Only the event creator can set a cover photo")
        elif usage_type == MediaUsageType.GALLERY:
            if not owner.allow_contributions:
                raise HTTPException(403, "This event does not allow contributions")

            check_event_media_limits(session, owner_id)

            # Ensure participant status
            if owner.created_by_id != current_user.id:
                participant_check = session.exec(
                    select(EventParticipant).where(
                        EventParticipant.event_id == owner_id,
                        EventParticipant.user_id == current_user.id,
                    )
                ).first()

                if not participant_check:
                    raise HTTPException(403, "You are not a participant of this event")
                elif participant_check.status != "approved":
                    raise HTTPException(403, "Your participation is not approved")
        else:
            raise HTTPException(400, f"Invalid usage_type for event: {usage_type}")
    else:
        raise HTTPException(400, "Invalid owner_type")

    # --- Validate file type ---
    if file.content_type:
        if file.content_type.startswith("image/"):
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    400,
                    f"Unsupported image type: {file.content_type}. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
                )
        elif file.content_type.startswith("video/"):
            if file.content_type not in ALLOWED_VIDEO_TYPES:
                raise HTTPException(
                    400,
                    f"Unsupported video type: {file.content_type}. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
                )
        else:
            raise HTTPException(400, "Only images and videos are supported.")
    else:
        raise HTTPException(400, "Could not determine file type")

    try:
        # --- Upload logic ---
        approval_status = "approved"
        if (
            owner_type == "event"
            and usage_type == MediaUsageType.GALLERY
            and owner.created_by_id != current_user.id
            and not owner.auto_approve_uploads
        ):
            approval_status = "pending"

        # Replace old cover if needed
        if usage_type == MediaUsageType.COVER_PHOTO:
            old_usage = session.exec(
                select(MediaUsage).where(
                    MediaUsage.owner_id == owner_id,
                    MediaUsage.owner_type == owner_type,
                    MediaUsage.usage_type == usage_type,
                )
            ).first()
            if old_usage:
                old_usage.usage_type = MediaUsageType.COVER_PHOTO_ARCHIVED
                session.add(old_usage)
                session.commit()

        # Upload to storage (e.g., ImageKit)
        uploaded_media = upload_file(
            file=file,
            user=current_user,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
        )

        # Save DB record
        saved_media = save_file_to_db(
            session=session,
            media=uploaded_media,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
            embeddings=None,
            user_id=current_user.id,
            approval_status=approval_status,
        )

        # Notify event creator
        if owner_type == "event" and usage_type == MediaUsageType.GALLERY:
            if current_user.id != owner.created_by_id:
                event = owner
                event_message = (
                    f"{current_user.full_name} uploaded media pending your approval."
                    if approval_status == "pending"
                    else f"{current_user.full_name} uploaded new media."
                )
                send_notification.delay(
                    user_id=event.created_by_id,
                    event="media_uploaded",
                    data={
                        "event_id": event.id,
                        "event_name": event.name,
                        "uploader_name": current_user.full_name,
                        "message": event_message,
                    },
                )

        # Generate embeddings in background (skip for cover photos)
        if usage_type != MediaUsageType.COVER_PHOTO:
            embed_media.delay(saved_media.id, uploaded_media.external_url)

        return SingleItemResponse(
            data=saved_media,
            message=(
                "Media uploaded successfully."
                if approval_status == "approved"
                else "Media uploaded successfully. Pending approval."
            ),
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
    status: Optional[str] = Query(
        None, description="Filter by approval status: pending, approved, rejected"
    ),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    """
    Get media for an event with optional filters.

    - Regular users: Only see approved media
    - Event creators: Can see all media, or filter by status
    """
    # Check if event exists
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Check if current user is the event creator
    is_event_creator = event.created_by_id == current_user.id

    # Event Privacy Guards: Restrict gallery access based on event privacy
    if event.privacy == "private":
        # Private events: only creator or approved participants can access gallery
        if not is_event_creator:
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

    # Validate status parameter
    valid_statuses = ["pending", "approved", "rejected"]
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}",
        )

    # Authorization check for status filter
    if status and not is_event_creator:
        # Non-creators can only see approved media
        if status != "approved":
            raise HTTPException(
                status_code=403,
                detail="Only event creators can view pending or rejected media",
            )

    # Base query: select Media and MediaUsage
    query = (
        select(Media, MediaUsage)
        .join(MediaUsage, MediaUsage.media_id == Media.id)
        .where(
            MediaUsage.owner_id == event_id,
            MediaUsage.owner_type == ContentOwnerType.EVENT,
            MediaUsage.usage_type == MediaUsageType.GALLERY,
        )
    )

    # Apply approval status filter
    if status:
        # Explicit status filter (only for event creators)
        query = query.where(MediaUsage.approval_status == status)
    else:
        # Default behavior based on user role
        if is_event_creator:
            # Event creator sees all media if no status specified
            pass
        else:
            # Non-creators only see approved media
            query = query.where(MediaUsage.approval_status == "approved")

    # Apply media type filter
    if media_type == "image":
        query = query.where(Media.mime_type.like("image/%"))
    elif media_type == "video":
        query = query.where(Media.mime_type.like("video/%"))

    # Count total
    total = session.exec(select(func.count()).select_from(query.subquery())).one()

    # Apply pagination
    offset = (page - 1) * per_page
    results = session.exec(
        query.order_by(Media.created_at.desc()).offset(offset).limit(per_page)
    ).all()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    # Convert to MediaRead objects with usage info
    media_reads = [media.to_media_read() for media, usage in results]

    # Build response message
    status_msg = f" ({status})" if status else ""
    role_msg = " (creator view)" if is_event_creator and not status else ""

    return PaginatedResponse[MediaRead](
        message=f"Event media{status_msg}{role_msg} retrieved successfully",
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
        # Delete media and associated files
        delete_media_and_file(session, media)
        session.commit()

        # Notify uploader if creator deleted their media
        if media.uploaded_by_id != current_user.id:
            send_notification.delay(
                user_id=media.uploaded_by_id,
                event="media_deleted",
                data={
                    "media_id": media.id,
                    "event_id": media_usage.owner_id,
                    "message": f"Your media in event '{event.name}' was removed by the creator.",
                },
            )

        return {"message": "Media deleted successfully", "media_id": media_id}

    except PermissionError as e:
        session.rollback()
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete media: {str(e)}")


@router.patch("/{media_id}/approve")
async def approve_media(
    media_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Approve a pending media upload (event creators only)"""

    media = session.get(Media, media_id)
    if not media:
        raise HTTPException(404, "Media not found")

    # Get media usage
    usage = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media_id)
    ).first()

    if not usage:
        raise HTTPException(404, "Media usage not found")

    # Only event gallery media needs approval
    if (
        usage.owner_type != ContentOwnerType.EVENT
        or usage.usage_type != MediaUsageType.GALLERY
    ):
        raise HTTPException(400, "This media does not require approval")

    # Check if user is event creator
    event = session.get(Event, usage.owner_id)
    if not event or event.created_by_id != current_user.id:
        raise HTTPException(403, "Only event creators can approve media")

    # Check current status
    if usage.approval_status == "approved":
        raise HTTPException(400, "Media is already approved")

    # Update approval status
    usage.approval_status = "approved"
    usage.approved_at = datetime.now(timezone.utc)
    session.commit()

    # Notify uploader
    send_notification.delay(
        user_id=media.uploaded_by_id,
        event="media_approved",
        data={
            "media_id": media.id,
            "event_id": event.id,
            "event_name": event.name,
            "message": f"Your media in '{event.name}' has been approved!",
        },
    )

    return SingleItemResponse(
        data=media.to_media_read(), message="Media approved successfully"
    )


@router.patch("/{media_id}/reject")
async def reject_media(
    media_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Reject a pending media upload (event creators only)"""

    media = session.get(Media, media_id)
    if not media:
        raise HTTPException(404, "Media not found")

    usage = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media_id)
    ).first()

    if not usage:
        raise HTTPException(404, "Media usage not found")

    if (
        usage.owner_type != ContentOwnerType.EVENT
        or usage.usage_type != MediaUsageType.GALLERY
    ):
        raise HTTPException(400, "This media does not require approval")

    event = session.get(Event, usage.owner_id)
    if not event or event.created_by_id != current_user.id:
        raise HTTPException(403, "Only event creators can reject media")

    if usage.approval_status == "rejected":
        raise HTTPException(400, "Media is already rejected")

    # Update rejection status
    usage.approval_status = "rejected"
    usage.approved_at = datetime.now(timezone.utc)
    session.commit()

    # Notify uploader
    send_notification.delay(
        user_id=media.uploaded_by_id,
        event="media_rejected",
        data={
            "data": {
                "media_id": media.id,
                "event_id": event.id,
                "event_name": event.name,
                "message": f"Your media in '{event.name}' has been rejected.",
            },
        },
    )

    return SingleItemResponse(data=media.to_media_read(), message="Media rejected")
