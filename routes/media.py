import os
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from sqlmodel import Session, func, select
from db import get_session
from models import User, Event, Media, SingleItemResponse
from models.core import PaginatedResponse, Pagination
from utils.users import get_current_user
from typing import Optional


router = APIRouter(prefix="/media", tags=["media"])

MEDIA_DIR = "static/media"
os.makedirs(MEDIA_DIR, exist_ok=True)

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}


@router.post("/", response_model=SingleItemResponse[Media])
async def upload_media(
    file: UploadFile = File(...),
    owner_id: int = Form(...),
    owner_type: str = Form(...),
    media_type: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    # Validate media_type
    if media_type not in ["image", "video"]:
        raise HTTPException(
            status_code=400, detail="media_type must be 'image' or 'video'"
        )

    # Validate MIME type
    content_type = file.content_type
    if media_type == "image" and content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type")
    if media_type == "video" and content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Invalid video type")

    # Validate owner exists based on owner_type
    owner = None
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

    # Save file to disk
    file_ext = os.path.splitext(file.filename)[1].lower()
    safe_filename = f"{owner_type}_{owner_id}_{media_type}{file_ext}"
    file_path = os.path.join(MEDIA_DIR, safe_filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Create a URL
    file_url = f"/{file_path}"

    # Save to DB
    media = Media(
        url=file_url,
        type=media_type,
        owner_id=owner_id,
        owner_type=owner_type,
        uploaded_by=current_user.id,
    )
    session.add(media)
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
        .filter(Media.uploaded_by == current_user.id)
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
    # Base query
    query = select(Media).where(
        Media.owner_id == event_id,
        Media.owner_type == "event",  # since you're using polymorphic owner
    )

    # Apply filter if media_type is specified
    if media_type:
        query = query.where(Media.type == media_type)

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
