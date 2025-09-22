import os
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from sqlmodel import Session, func, select
from db import get_session
from models import (
    User,
    Media,
    MediaRead,
    MediaUsage,
    Event,
    ContentOwnerType,
    MediaUsageType,
    PaginatedResponse,
    Pagination,
    SingleItemResponse,
)
from utils.users import get_current_user
from typing import Optional
from dotenv import load_dotenv
from utils.face import (
    delete_media_and_file,
    delete_old_face_enrollment,
    generate_face_embeddings,
    get_drive_service,
)
from models import MediaEmbedding
from googleapiclient.http import MediaIoBaseDownload
from utils.media import upload_file, save_file_to_db


load_dotenv()

FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME")
FACE_MODEL_BACKEND = os.environ.get("FACE_MODEL_BACKEND")

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}

router = APIRouter(prefix="/uploads", tags=["media"])


@router.post("", response_model=SingleItemResponse[Media])
async def upload_media(
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
    # --- validate owner ---
    if owner_type == "user":
        owner = session.get(User, owner_id)
    elif owner_type == "event":
        owner = session.get(Event, owner_id)
    else:
        raise HTTPException(400, "Invalid owner_type")

    if not owner:
        raise HTTPException(404, f"{owner_type.capitalize()} not found")

    supported_usage_types = [item.value for item in MediaUsageType]
    if usage_type not in supported_usage_types:
        raise HTTPException(400, f"Unsupported usage_type: {usage_type}")

    try:
        # Generate embeddings first (this will read the file)
        embeddings = generate_face_embeddings(file)

        # Upload file
        uploaded_media = upload_file(
            file=file,
            user=current_user,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
        )

        # Save to database
        saved_media = save_file_to_db(
            session=session,
            media=uploaded_media,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
            embeddings=embeddings,
            user_id=current_user.id,
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
    total = session.exec(
        select(func.count(Media.id)).where(Media.uploaded_by_id == current_user.id)
    ).one()
    offset = (page - 1) * per_page
    user_uploads = session.exec(
        select(Media)
        .where(Media.uploaded_by_id == current_user.id)
        .order_by(Media.created_at.desc())
        .offset(offset)
        .limit(per_page)
    ).all()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[MediaRead](
        message="User uploads retrieved successfully",
        data=user_uploads,
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

    # Convert to MediaRead with base64 data
    drive_service = get_drive_service(current_user)
    media_reads = []
    for media in event_media:
        media_reads.append(media)

    total_pages = ((total - 1) // per_page) + 1 if total else 0

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
