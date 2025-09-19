import os
import tempfile
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
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
    save_media_file,
    get_drive_service,
)
from models import MediaEmbedding
from googleapiclient.http import MediaIoBaseDownload


load_dotenv()


MEDIA_DIR = os.environ.get("MEDIA_DIR")
os.makedirs(MEDIA_DIR, exist_ok=True)

FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME")
FACE_MODEL_BACKEND = os.environ.get("FACE_MODEL_BACKEND")
MEDIA_DIR = os.environ.get("MEDIA_DIR")

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
    Uploads a file to Google Drive for the user.
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
        # --- Step 1: save media (Drive upload + DB insert) ---
        media = save_media_file(
            session=session,
            file=file,
            user=current_user,
            owner_id=owner_id,
            owner_type=owner_type,
            usage_type=usage_type,
        )

        # --- Step 2: embeddings only for images ---
        if media.mime_type.startswith("image/"):
            # download file back from Drive to temp for embedding
            drive_service = get_drive_service(current_user)
            local_fd, local_path = tempfile.mkstemp(suffix=".jpg")
            os.close(local_fd)

            request = drive_service.files().get_media(fileId=media.external_id)
            with open(local_path, "wb") as tmp:
                downloader = MediaIoBaseDownload(tmp, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            try:
                embeddings = generate_face_embeddings(
                    local_path, media.mime_type, current_user
                )
            except Exception as e:
                delete_media_and_file(session, media, current_user)
                session.commit()
                raise HTTPException(500, f"Face processing failed: {str(e)}")
            finally:
                if os.path.exists(local_path):
                    os.remove(local_path)

            # if this is enrollment, remove any old enrollment embeddings
            if usage_type == "face_enrollment":
                delete_old_face_enrollment(session, current_user.id)

            # save embeddings (one per detected face)
            for emb in embeddings:
                media_embedding = MediaEmbedding(
                    media_id=media.id,
                    model_name=FACE_MODEL_NAME,
                    user_id=(
                        current_user.id if usage_type == "face_enrollment" else None
                    ),
                )
                media_embedding.embedding = emb
                session.add(media_embedding)

        # --- Step 3: commit ---
        session.commit()
        session.refresh(media)
        return SingleItemResponse(data=media, message="Media uploaded successfully")

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

    # Convert to MediaRead with base64 data
    drive_service = get_drive_service(current_user)
    media_reads = []
    for media in user_uploads:
        media_reads.append(media.to_media_read(current_user, drive_service))

    total_pages = ((total - 1) // per_page) + 1 if total else 0

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
        media_reads.append(media.to_media_read(current_user, drive_service))

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
