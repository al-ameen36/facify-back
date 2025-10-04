from fastapi import APIRouter, Depends
from db import get_session
from sqlmodel import select, Session, func
from models import User, UnknownFaceCluster, FaceMatch, Media


router = APIRouter(prefix="/faces", tags=["face"])


@router.get("/gallery")
def list_face_groups(
    event_id: int,
    page: int = 1,
    per_page: int = 10,
    session: Session = Depends(get_session),
):
    """
    List all recognized faces (matched users) and clustered unknown faces for an event.
    """
    # Get known users (matched faces)
    known_users_query = (
        select(User)
        .join(FaceMatch, FaceMatch.matched_user_id == User.id)
        .where(
            FaceMatch.event_id == event_id,
            FaceMatch.matched_user_id.is_not(None),
        )
        .distinct()
    )
    total_known = session.exec(
        select(func.count()).select_from(known_users_query.subquery())
    ).one()
    known_users = session.exec(
        known_users_query.offset((page - 1) * per_page).limit(per_page)
    ).all()

    # Serialize known users with profile picture
    data = []
    for user in known_users:
        profile = user.get_profile_picture_media(session)
        data.append(
            {
                "id": user.id,
                "username": user.username,
                "profile_picture": profile.model_dump() if profile else None,
            }
        )

    # Get unknown clusters
    unknown_clusters = session.exec(
        select(UnknownFaceCluster.cluster_label).where(
            UnknownFaceCluster.event_id == event_id
        )
    ).all()

    return {
        "message": "Event gallery retrieved successfully",
        "data": {
            "known_faces": data,
            "unknown_clusters": [c.cluster_label for c in unknown_clusters],
        },
        "pagination": {
            "total": total_known,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_known + per_page - 1) // per_page,
        },
    }


@router.get("/events/{event_id}/users/{user_id}/images")
def get_user_event_images(
    event_id: int,
    user_id: int,
    page: int = 1,
    per_page: int = 10,
    session: Session = Depends(get_session),
):
    """
    Get all images for a specific user matched in an event.
    """
    query = (
        select(Media)
        .join(FaceMatch, FaceMatch.media_id == Media.id)
        .where(
            FaceMatch.event_id == event_id,
            FaceMatch.matched_user_id == user_id,
            Media.mime_type.like("image/%"),
        )
        .distinct()
    )

    total = session.exec(select(func.count()).select_from(query.subquery())).one()

    medias = session.exec(query.offset((page - 1) * per_page).limit(per_page)).all()

    data = [m.to_media_read().model_dump() for m in medias]

    return {
        "message": "User's event media retrieved successfully",
        "data": data,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        },
    }
