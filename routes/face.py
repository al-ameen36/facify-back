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
    thumbnail: Optional[str] = None
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
    thumbnail: Optional[str] = None
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

                # --- Manual Face Crop ---
                PADDING_RATIO = 0.40
                if fe.facial_area:
                    fa = fe.facial_area
                    x, y, w, h = fa.get("x"), fa.get("y"), fa.get("w"), fa.get("h")

                    if None not in (x, y, w, h):
                        # Add padding
                        pad_w = int(w * PADDING_RATIO)
                        pad_h = int(h * PADDING_RATIO)

                        new_x = max(x - pad_w, 0)
                        new_y = max(y - pad_h, 0)
                        new_w = w + 2 * pad_w
                        new_h = h + 2 * pad_h

                        thumb_url = (
                            f"{media_read.url}"
                            f"?tr=x-{new_x},y-{new_y},w-{new_w},h-{new_h},cm-extract,"
                            f"w-400,h-400,c-force,q-80"
                        )
                    else:
                        thumb_url = (
                            f"{media_read.url}?tr=w-400,h-400,c-maintain_ratio,q-80"
                        )

                # Append cluster media
                cluster_media_map[fe.cluster_id].append(
                    ClusterMediaItem(
                        id=media_read.id,
                        url=media_read.url,
                        thumbnail=thumb_url,
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

        # Pick the best thumbnail (largest face area)
        thumbnail_url = None
        if media_list:
            largest = max(
                media_list,
                key=lambda m: (
                    (m.face_area.get("w", 0) * m.face_area.get("h", 0))
                    if m.face_area
                    else 0
                ),
            )
            thumbnail_url = largest.thumbnail

        gallery.append(
            ClusterGalleryItem(
                cluster_id=cluster.id,
                label=cluster.label or f"Person {cluster.id}",
                user=user_info,
                face_count=len(media_list),
                thumbnail=thumbnail_url,
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


@router.get(
    "/clusters/{cluster_id}", response_model=PaginatedResponse[ClusterGalleryItem]
)
async def get_cluster_faces(
    cluster_id: int,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    IMAGEKIT_TRANSFORM_BASE = "w-400,h-400,c-force,q-80"
    PADDING_RATIO = 0.25

    # --- Step 1: Get cluster ---
    cluster = session.exec(
        select(FaceCluster)
        .where(FaceCluster.id == cluster_id)
        .options(selectinload(FaceCluster.user))
    ).first()

    if not cluster:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # --- Step 2: Get embeddings for this cluster ---
    embeddings_stmt = (
        select(FaceEmbedding)
        .where(FaceEmbedding.cluster_id == cluster_id)
        .options(selectinload(FaceEmbedding.media))
    )
    embeddings = session.exec(embeddings_stmt).all()

    if not embeddings:
        return PaginatedResponse[ClusterGalleryItem](
            message="No faces found for this cluster",
            data=[],
            pagination=Pagination(total=0, page=page, per_page=per_page, total_pages=0),
        )

    # --- Step 3: Pagination ---
    total = len(embeddings)
    offset = (page - 1) * per_page
    paginated_embeddings = embeddings[offset : offset + per_page]
    total_pages = ((total - 1) // per_page) + 1 if total else 0

    # --- Step 4: Build media list ---
    media_list = []
    for fe in paginated_embeddings:
        if not fe.media:
            continue

        media = fe.media.to_media_read()
        face_area = fe.facial_area or {}

        # compute padded crop
        thumb_url = media.url
        if {"x", "y", "w", "h"} <= face_area.keys():
            x, y, w, h = (
                face_area["x"],
                face_area["y"],
                face_area["w"],
                face_area["h"],
            )
            pad_w = int(w * PADDING_RATIO)
            pad_h = int(h * PADDING_RATIO)
            new_x = max(x - pad_w, 0)
            new_y = max(y - pad_h, 0)
            new_w = w + 2 * pad_w
            new_h = h + 2 * pad_h
            thumb_url = (
                f"{media.url}"
                f"?tr=x-{new_x},y-{new_y},w-{new_w},h-{new_h},cm-extract,{IMAGEKIT_TRANSFORM_BASE}"
            )

        media_list.append(
            ClusterMediaItem(
                id=media.id,
                url=media.url,
                thumbnail=thumb_url,
                filename=media.filename,
                mime_type=media.mime_type,
                file_size=media.file_size,
                duration=media.duration,
                uploaded_at=(
                    media.created_at.isoformat()
                    if hasattr(media.created_at, "isoformat")
                    else media.created_at
                ),
                face_area=face_area,
                status=media.status,
            )
        )

    # --- Step 5: Cluster user info ---
    user_info = None
    if cluster.user:
        profile_pic = cluster.user.get_profile_picture_media(session)
        user_info = ClusterUserInfo(
            id=cluster.user.id,
            full_name=cluster.user.full_name,
            username=cluster.user.username,
            profile_picture=profile_pic.url if profile_pic else None,
        )

    # --- Step 6: Build response item ---
    item = ClusterGalleryItem(
        cluster_id=cluster.id,
        label=cluster.label or f"Person {cluster.id}",
        user=user_info,
        face_count=len(embeddings),
        thumbnail=None,  # string thumbnail is now inside media items
        media=media_list,
    )

    # --- Step 7: Return standard paginated response ---
    return PaginatedResponse[ClusterGalleryItem](
        message=f"Faces retrieved successfully for cluster {cluster_id}",
        data=[item],
        pagination=Pagination(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        ),
    )


@router.get("/user/{user_id}", response_model=PaginatedResponse[ClusterGalleryItem])
async def get_user_clusters(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session=Depends(get_session),
    skip_auth: bool = Query(False, description="Skip authorization (for debugging)"),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    """
    Return paginated clusters that are linked to `user_id` (clusters are event-scoped).
    Each cluster includes media items (face thumbnails generated from facial_area).
    """

    # Authorization: only allow requesting other user's clusters if skip_auth
    if (
        not skip_auth
        and current_user.id != user_id
        and not getattr(current_user, "is_admin", False)
    ):
        raise HTTPException(
            status_code=403, detail="Not authorized to view this user's clusters"
        )

    # --- Step 1: Find clusters that belong to the user_id ---
    clusters_stmt = (
        select(FaceCluster)
        .where(FaceCluster.user_id == user_id)
        .options(selectinload(FaceCluster.user))
    )
    clusters_all = session.exec(clusters_stmt).all()
    if not clusters_all:
        return PaginatedResponse[ClusterGalleryItem](
            message="No clusters found for this user",
            data=[],
            pagination=Pagination(total=0, page=page, per_page=per_page, total_pages=0),
        )

    # --- Step 2: Pagination at cluster level ---
    total = len(clusters_all)
    offset = (page - 1) * per_page
    paginated_clusters = clusters_all[offset : offset + per_page]
    total_pages = ((total - 1) // per_page) + 1 if total else 0

    # --- Step 3: For these clusters, fetch embeddings and media ---
    cluster_ids = [c.id for c in paginated_clusters]

    embeddings_stmt = (
        select(FaceEmbedding)
        .where(FaceEmbedding.cluster_id.in_(cluster_ids))
        .options(selectinload(FaceEmbedding.media))
    )
    embeddings = session.exec(embeddings_stmt).all()

    # Build a map cluster_id -> list of embeddings
    embeddings_by_cluster = {}
    for fe in embeddings:
        embeddings_by_cluster.setdefault(fe.cluster_id, []).append(fe)

    # ImageKit cropping params
    IMAGEKIT_TRANSFORM_BASE = "w-400,h-400,c-force,q-80"
    PADDING_RATIO = 0.25

    # --- Step 4: Build gallery items ---
    gallery_items: List[ClusterGalleryItem] = []
    for cluster in paginated_clusters:
        embs = embeddings_by_cluster.get(cluster.id, [])
        media_items: List[ClusterMediaItem] = []

        for fe in embs:
            if not fe.media:
                continue
            try:
                media_read = fe.media.to_media_read()
            except Exception:
                # fallback if to_media_read fails
                continue

            face_area = fe.facial_area or {}
            thumb_url = media_read.url

            # If we have bounding box, apply padding and use cm-extract
            if {"x", "y", "w", "h"} <= set(face_area.keys()):
                x = int(face_area["x"])
                y = int(face_area["y"])
                w = int(face_area["w"])
                h = int(face_area["h"])

                pad_w = int(w * PADDING_RATIO)
                pad_h = int(h * PADDING_RATIO)
                new_x = max(x - pad_w, 0)
                new_y = max(y - pad_h, 0)
                new_w = w + 2 * pad_w
                new_h = h + 2 * pad_h

                thumb_url = (
                    f"{media_read.url}"
                    f"?tr=x-{new_x},y-{new_y},w-{new_w},h-{new_h},cm-extract,{IMAGEKIT_TRANSFORM_BASE}"
                )

            media_items.append(
                ClusterMediaItem(
                    id=media_read.id,
                    url=media_read.url,
                    thumbnail=thumb_url,
                    filename=media_read.filename,
                    mime_type=media_read.mime_type,
                    file_size=media_read.file_size,
                    duration=media_read.duration,
                    uploaded_at=(
                        media_read.created_at.isoformat()
                        if hasattr(media_read.created_at, "isoformat")
                        else media_read.created_at
                    ),
                    face_area=face_area,
                    status=media_read.status,
                )
            )

        # pick best thumbnail (largest face area) as a string URL
        thumbnail_url: Optional[str] = None
        if media_items:
            largest = max(
                media_items,
                key=lambda m: (
                    (m.face_area.get("w", 0) * m.face_area.get("h", 0))
                    if m.face_area
                    else 0
                ),
            )
            thumbnail_url = largest.thumbnail

        # cluster user info (should be the same user_id for all these clusters)
        user_info = None
        if cluster.user:
            profile_pic = cluster.user.get_profile_picture_media(session)
            user_info = ClusterUserInfo(
                id=cluster.user.id,
                full_name=cluster.user.full_name,
                username=cluster.user.username,
                profile_picture=profile_pic.url if profile_pic else None,
            )

        gallery_items.append(
            ClusterGalleryItem(
                cluster_id=cluster.id,
                label=cluster.label or f"Person {cluster.id}",
                user=user_info,
                face_count=len(embs),
                thumbnail=thumbnail_url,
                media=media_items,
            )
        )

    # Optionally sort clusters by face_count (most faces first)
    gallery_items.sort(key=lambda x: x.face_count, reverse=True)

    # --- Step 5: Return using standard paginated response ---
    return PaginatedResponse[ClusterGalleryItem](
        message=f"Clusters retrieved successfully for user {user_id}",
        data=gallery_items,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
        ),
    )
