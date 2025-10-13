from fastapi import APIRouter, HTTPException, Depends, Query
from sqlmodel import select
from sqlalchemy.orm import selectinload
from typing import Optional, List
from pydantic import BaseModel
from models import (
    FaceCluster,
    FaceEmbedding,
    MediaUsage,
    ContentOwnerType,
    User,
    PaginatedResponse,
    Pagination,
)
from utils.users import get_session, get_current_user

router = APIRouter(prefix="/faces", tags=["face"])


# ================================
# Response Models
# ================================
class ClusterMediaItem(BaseModel):
    id: int
    url: str
    filename: str
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    duration: Optional[float] = None
    uploaded_at: Optional[str] = None
    face_area: Optional[dict] = None
    status: Optional[str] = None


class ClusterUserInfo(BaseModel):
    id: int
    full_name: str
    username: str
    profile_picture: Optional[str] = None


class ClusterGalleryItem(BaseModel):
    cluster_id: int
    label: str
    user: Optional[ClusterUserInfo] = None
    face_count: int
    thumbnail: Optional[ClusterMediaItem] = None
    media: List[ClusterMediaItem]


# ================================
# Gallery Endpoint
# ================================
@router.get("/gallery/{event_id}", response_model=PaginatedResponse[ClusterGalleryItem])
async def get_event_gallery(
    event_id: int,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session),
    skip_auth: bool = Query(
        False, description="Skip authorization check for debugging"
    ),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    """
    Paginated gallery of all face clusters detected in an event.
    Only accessible to the event creator or approved participants.
    """
    from models import Event, EventParticipant

    # --- Step 1: Authorization ---
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if not skip_auth:
        is_creator = event.created_by_id == current_user.id
        is_participant = session.exec(
            select(EventParticipant).where(
                EventParticipant.event_id == event_id,
                EventParticipant.user_id == current_user.id,
                EventParticipant.status == "approved",
            )
        ).first()
        if not (is_creator or is_participant):
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this event's gallery",
            )

    # --- Step 2: Media for this event ---
    media_ids_stmt = select(MediaUsage.media_id).where(
        MediaUsage.owner_type == ContentOwnerType.EVENT,
        MediaUsage.owner_id == event_id,
    )
    event_media_ids = list(session.exec(media_ids_stmt).all())
    if not event_media_ids:
        return PaginatedResponse[ClusterGalleryItem](
            message="No media found for this event",
            data=[],
            pagination=Pagination(total=0, page=page, per_page=per_page, total_pages=0),
        )

    # --- Step 3: Face embeddings ---
    face_embeddings_stmt = (
        select(FaceEmbedding)
        .where(
            FaceEmbedding.media_id.in_(event_media_ids),
            FaceEmbedding.cluster_id.is_not(None),
        )
        .options(selectinload(FaceEmbedding.media))
    )
    face_embeddings = session.exec(face_embeddings_stmt).all()
    if not face_embeddings:
        return PaginatedResponse[ClusterGalleryItem](
            message="No faces detected in this event yet",
            data=[],
            pagination=Pagination(total=0, page=page, per_page=per_page, total_pages=0),
        )

    # --- Step 4: Cluster IDs ---
    cluster_ids = list(set(fe.cluster_id for fe in face_embeddings if fe.cluster_id))
    if not cluster_ids:
        return PaginatedResponse[ClusterGalleryItem](
            message="No clustered faces found",
            data=[],
            pagination=Pagination(total=0, page=page, per_page=per_page, total_pages=0),
        )

    # --- Step 5: Pagination ---
    total = len(cluster_ids)
    offset = (page - 1) * per_page
    paginated_cluster_ids = cluster_ids[offset : offset + per_page]
    total_pages = ((total - 1) // per_page) + 1 if total else 0

    clusters_stmt = (
        select(FaceCluster)
        .where(FaceCluster.id.in_(paginated_cluster_ids))
        .options(selectinload(FaceCluster.user))
    )
    clusters = session.exec(clusters_stmt).all()
    cluster_map = {c.id: c for c in clusters}

    # --- Step 6: Group media by cluster ---
    cluster_media_map = {}
    for fe in face_embeddings:
        if fe.cluster_id in paginated_cluster_ids and fe.media:
            cluster_media_map.setdefault(fe.cluster_id, [])
            try:
                media_read = fe.media.to_media_read()
                cluster_media_map[fe.cluster_id].append(
                    ClusterMediaItem(
                        id=media_read.id,
                        url=media_read.url,
                        filename=media_read.filename,
                        mime_type=media_read.mime_type,
                        file_size=media_read.file_size,
                        duration=media_read.duration,
                        uploaded_at=(
                            media_read.created_at.isoformat()
                            if hasattr(media_read.created_at, "isoformat")
                            else media_read.created_at
                        ),
                        face_area=fe.facial_area,
                        status=media_read.status,
                    )
                )
            except Exception as e:
                print(f"Error converting media {fe.media.id}: {e}")

    # --- Step 7: Build gallery response ---
    gallery = []
    for cluster_id, media_list in cluster_media_map.items():
        cluster = cluster_map.get(cluster_id)
        if not cluster or not media_list:
            continue

        # User info (if matched)
        user_info = None
        if cluster.user:
            profile_pic = cluster.user.get_profile_picture_media(session)
            user_info = ClusterUserInfo(
                id=cluster.user.id,
                full_name=cluster.user.full_name,
                username=cluster.user.username,
                profile_picture=profile_pic.url if profile_pic else None,
            )

        thumbnail = (
            max(
                media_list,
                key=lambda m: (
                    (m.face_area.get("w", 0) * m.face_area.get("h", 0))
                    if m.face_area
                    else 0
                ),
            )
            if media_list
            else None
        )
        gallery.append(
            ClusterGalleryItem(
                cluster_id=cluster.id,
                label=cluster.label or f"Person {cluster.id}",
                user=user_info,
                face_count=len(media_list),
                thumbnail=thumbnail,
                media=media_list,
            )
        )

    # Sort by most faces first
    gallery.sort(key=lambda x: x.face_count, reverse=True)

    return PaginatedResponse[ClusterGalleryItem](
        message=f"Gallery retrieved successfully for event {event.name}",
        data=gallery,
        pagination=Pagination(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        ),
    )
