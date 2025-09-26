import os
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException, BackgroundTasks
from sqlmodel import Session, func, select
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
from utils.users import get_current_user
from utils.face import (
    generate_face_embeddings,
)
from utils.media import upload_file, save_file_to_db, delete_media_and_file
from models import MediaEmbedding



load_dotenv()

FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME")
FACE_MODEL_BACKEND = os.environ.get("FACE_MODEL_BACKEND")
FACE_API_URL = os.environ.get("FACE_API_URL")

# Allowed MIME types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}

router = APIRouter(prefix="/uploads", tags=["media"])


@router.post("", response_model=Union[SingleItemResponse[Media], SingleItemResponse[UserRead]])
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
        
        # For events, check authorization based on usage type
        if usage_type == MediaUsageType.COVER_PHOTO:
            # Only event creator can upload cover photos
            if owner.created_by_id != current_user.id:
                raise HTTPException(403, "Only event creators can upload cover photos")
        elif usage_type == MediaUsageType.GALLERY:
            # Event creator or approved participants can upload to gallery
            if owner.created_by_id != current_user.id:
                participant_check = session.exec(
                    select(EventParticipant).where(
                        EventParticipant.event_id == owner_id,
                        EventParticipant.user_id == current_user.id
                    )
                ).first()
                
                if not participant_check:
                    raise HTTPException(403, "You are not a participant of this event")
                elif participant_check.status == "pending":
                    raise HTTPException(403, "Your participation request is still pending approval")
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
                delete_media_and_file(session, old_usage.media, current_user)
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
            background_tasks.add_task(
                generate_embeddings_background,
                saved_media.id,
                uploaded_media.external_url,
                uploaded_media.external_id
            )

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
    
    # Check if user is event creator or approved participant
    if event.created_by_id != current_user.id:
        participant_check = session.exec(
            select(EventParticipant).where(
                EventParticipant.event_id == event_id,
                EventParticipant.user_id == current_user.id
            )
        ).first()
        
        if not participant_check:
            raise HTTPException(status_code=403, detail="You are not a participant of this event")
        elif participant_check.status == "pending":
            raise HTTPException(status_code=403, detail="Your participation request is still pending approval")
        elif participant_check.status == "rejected":
            raise HTTPException(status_code=403, detail="Your participation request was rejected")
        elif participant_check.status != "approved":
            raise HTTPException(status_code=403, detail="Access denied")
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


def generate_embeddings_background(media_id: int, external_url: str, external_id: str):
    """Background task to generate face embeddings for uploaded media"""
    from db import get_session
    import requests
    from datetime import datetime
    
    # Validate configuration
    if not FACE_API_URL:
        print(f"FACE_API_URL not configured, skipping embeddings for media_id {media_id}")
        return
    
    session = next(get_session())
    
    try:
        # Get the media embedding record
        media_embedding = session.exec(
            select(MediaEmbedding).where(MediaEmbedding.media_id == media_id)
        ).first()
        
        if not media_embedding:
            print(f"No MediaEmbedding found for media_id {media_id}")
            return
            
        # Update status to processing
        media_embedding.status = "processing"
        session.commit()
        
        # Download the image from external URL with timeout
        response = requests.get(external_url, timeout=30)
        response.raise_for_status()
        
        # Call face API with timeout
        face_response = requests.post(
            f"{FACE_API_URL}/embed",
            files={"file": ("image", response.content, "image/jpeg")},
            timeout=60
        )
        face_response.raise_for_status()
        results = face_response.json()
        
        # Extract embeddings
        embeddings = []
        if isinstance(results, list):
            for r in results:
                if "embedding" in r:
                    embeddings.append(r["embedding"])
        
        # Update the embedding record
        media_embedding.embeddings = embeddings
        media_embedding.status = "completed" if embeddings else "failed"
        media_embedding.processed_at = datetime.utcnow()
        if not embeddings:
            media_embedding.error_message = "No faces detected"
            
        session.commit()
        print(f"Successfully processed embeddings for media_id {media_id}: {len(embeddings)} faces found")
        
    except requests.RequestException as e:
        # Handle network errors specifically
        if 'media_embedding' in locals():
            media_embedding.status = "failed"
            media_embedding.error_message = f"Network error: {str(e)}"
            media_embedding.processed_at = datetime.utcnow()
            session.commit()
        print(f"Network error processing embeddings for media_id {media_id}: {str(e)}")
    except Exception as e:
        # Update status to failed
        if 'media_embedding' in locals():
            media_embedding.status = "failed"
            media_embedding.error_message = str(e)
            media_embedding.processed_at = datetime.utcnow()
            session.commit()
        print(f"Failed to process embeddings for media_id {media_id}: {str(e)}")
    
    finally:
        session.close()


@router.get("/embedding-status", response_model=dict)
def get_embedding_status(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get embedding processing status for debugging"""
    if not current_user.is_admin:
        raise HTTPException(403, "Admin access required")
    
    status_counts = session.exec(
        select(MediaEmbedding.status, func.count(MediaEmbedding.id))
        .group_by(MediaEmbedding.status)
    ).all()
    
    return {
        "embedding_status": {status: count for status, count in status_counts},
        "message": "Embedding processing status retrieved"
    }
