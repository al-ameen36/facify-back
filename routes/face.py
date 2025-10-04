from fastapi import APIRouter, Depends
from db import get_session
from sqlmodel import select, Session
from models import User, UnknownFaceCluster, FaceMatch


router = APIRouter(prefix="/faces", tags=["face"])


@router.get("/gallery")
def list_face_groups(event_id: int, session: Session = Depends(get_session)):
    """
    List all recognized faces (matched users) and clustered unknown faces for an event.
    Includes each user's profile picture (if available).
    """

    # Get all distinct users matched in this event
    known_users = session.exec(
        select(User)
        .join(FaceMatch, FaceMatch.matched_user_id == User.id)
        .where(
            FaceMatch.event_id == event_id,
            FaceMatch.matched_user_id.is_not(None),
        )
        .distinct()
    ).all()

    # Get unknown face clusters
    unknown_clusters = session.exec(
        select(UnknownFaceCluster.cluster_label).where(
            UnknownFaceCluster.event_id == event_id
        )
    ).all()

    # Build response with profile picture
    known_faces = []
    for user in known_users:
        profile_pic = user.get_profile_picture_media(session)
        known_faces.append(
            {
                "id": user.id,
                "username": user.username,
                "profile_picture": profile_pic.model_dump() if profile_pic else None,
            }
        )

    return {
        "known_faces": known_faces,
        "unknown_clusters": [c.cluster_label for c in unknown_clusters],
    }
