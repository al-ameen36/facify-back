import os
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlmodel import Session
from db import get_session
from models import User, Event, Media, SingleItemResponse
from utils.users import get_current_user

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
    user: User = Depends(get_current_user),
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
        uploaded_by=user.id,
    )
    session.add(media)
    session.commit()
    session.refresh(media)

    return SingleItemResponse(data=media, message="Media uploaded successfully")
