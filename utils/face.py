import numpy as np
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from sqlmodel import Session, select
from db import get_session
from models import (
    Media,
    MediaEmbedding,
    MediaUsage,
    UnknownFaceCluster,
    Event,
    EventParticipant,
    User,
    FaceMatch,
    ContentOwnerType,
    MediaUsageType,
)
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tasks.notifications import send_ws_notification_task
from fastapi.encoders import jsonable_encoder
from socket_io import sio


load_dotenv()

FACE_API_URL = os.environ.get("FACE_API_URL")
APP_NAME = os.environ.get("APP_NAME")


# Media ops
class NetworkError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(NetworkError),
    reraise=True,
)
def safe_request(method: str, url: str, **kwargs):
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
        resp.raise_for_status()
        return resp
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
        raise NetworkError(str(e))


async def generate_embeddings_background(
    session: Session, media_id: int, image_url: str
):
    media = session.get(Media, media_id)
    if not media:
        print(f"Media {media_id} not found")
        return

    embedding = session.exec(
        select(MediaEmbedding).where(MediaEmbedding.media_id == media.id)
    ).first()
    if not embedding:
        embedding = MediaEmbedding(
            media_id=media.id,
            model_name="ArcFace",
            status="processing",
        )
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

    else:
        # If embedding exists, update status to processing
        embedding.status = "processing"
        embedding.error_message = None
        session.commit()

    try:
        # Fetch image from ImageKit
        res = safe_request("GET", image_url)

        # Send to DeepFace / ArcFace server
        response = safe_request(
            "POST",
            f"{FACE_API_URL}/embed",
            files={"file": ("image.jpg", res.content, "image/jpeg")},
        )

        data = response.json()

        # Process response
        if "error" in data:
            embedding.status = "failed"
            embedding.error_message = data["error"]
            session.commit()

            await sio.emit(
                "notification",
                {
                    "type": "embedding_failed",
                    "media_id": media.id,
                    "error": data["error"],
                },
                room=f"user:{media.uploaded_by_id}",
            )
            return

        embeddings = (
            [face["embedding"] for face in data] if isinstance(data, list) else []
        )
        embedding.embeddings = embeddings
        embedding.status = "completed"
        embedding.processed_at = datetime.utcnow().isoformat()
        media.face_count = len(embedding.embeddings)
        session.commit()

        # Notify user: embedding done
        await sio.emit(
            "notification",
            {
                "type": "embedding_completed",
                "media_id": media.id,
                "count": len(embeddings),
            },
            room=f"user:{media.uploaded_by_id}",
        )

        # Trigger matching
        await match_faces_background(session, media.id, embedding.id)

    except Exception as e:
        embedding.status = "failed"
        embedding.error_message = str(e)
        session.commit()

        await sio.emit(
            "notification",
            {
                "type": "embedding_failed",
                "media_id": media.id,
                "error": str(e),
            },
            room=f"user:{media.uploaded_by_id}",
        )


async def match_faces_background(
    session: Session, media_id: int, embedding_id: int, threshold: float = 0.6
):
    """Match detected faces in an uploaded media against user embeddings and notify matched users."""
    # Load media embedding
    media_embedding = session.get(MediaEmbedding, embedding_id)
    if not media_embedding or not media_embedding.embeddings:
        print(f"[FaceMatch] No embeddings found for MediaEmbedding {embedding_id}")
        return

    # Load media + usage
    media = session.get(Media, media_id)
    if not media:
        print(f"[FaceMatch] Media {media_id} not found")
        return

    usage = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media_id)
    ).first()
    if not usage:
        print(f"[FaceMatch] No usage record found for media {media_id}")
        return

    # Skip face matching for non-event media (e.g., profile pictures)
    if (
        usage.owner_type != ContentOwnerType.EVENT
        or usage.usage_type != MediaUsageType.GALLERY
    ):
        print(
            f"[FaceMatch] Skipping face matching for non-event media {media_id} (owner_type={usage.owner_type}, usage_type={usage.usage_type})"
        )
        return

    # Load event
    event = session.get(Event, usage.owner_id)
    if not event:
        print(f"[FaceMatch] No event found for usage owner {usage.owner_id}")
        return

    # Get approved participants
    participant_ids = [
        ep.user_id
        for ep in session.exec(
            select(EventParticipant.user_id).where(
                EventParticipant.event_id == event.id,
                EventParticipant.status == "approved",
            )
        ).all()
    ]

    # Get all users with completed embeddings (profile pictures)
    users_with_embeddings = session.exec(
        select(User, MediaEmbedding)
        .join(MediaUsage, MediaUsage.owner_id == User.id)
        .join(Media, Media.id == MediaUsage.media_id)
        .join(MediaEmbedding, MediaEmbedding.media_id == Media.id)
        .where(
            MediaUsage.owner_type == ContentOwnerType.USER,
            MediaUsage.usage_type.in_(
                [
                    MediaUsageType.PROFILE_PICTURE,
                    MediaUsageType.PROFILE_PICTURE_ARCHIVED,
                ]
            ),
            MediaEmbedding.status == "completed",
        )
    ).all()

    if not users_with_embeddings:
        print("[FaceMatch] No users with valid embeddings found.")
        return

    matched_users = set()

    # Match embeddings
    for idx, emb in enumerate(media_embedding.embeddings or []):
        best_user, best_distance = None, 1.0
        for user, user_embedding in users_with_embeddings:
            if not user_embedding.embeddings:
                continue
            dist = cosine(emb, user_embedding.embeddings[0])
            if dist < best_distance:
                best_distance = dist
                best_user = user
        is_match = best_user is not None and best_distance < threshold
        match = FaceMatch(
            event_id=event.id,
            media_id=media_id,
            embedding_index=idx,
            matched_user_id=best_user.id if is_match else None,
            distance=float(best_distance),
            is_participant=(best_user.id in participant_ids) if is_match else False,
        )
        session.add(match)
        if is_match:
            matched_users.add(best_user.id)

    session.commit()

    # Prepare media payload (safe for WS)
    media_payload = jsonable_encoder(media)

    # Notify event owner (summary)
    await sio.emit(
        "notification",
        {
            "type": "face_matching_completed",
            "media": media_payload,
            "event_id": event.id,
            "matched_count": len(matched_users),
        },
        room=f"user:{event.created_by_id}",
    )

    # Notify each matched user
    for uid in matched_users:
        await sio.emit(
            "notification",
            {
                "type": "face_match_batch",
                "event_id": event.id,
                "media": media_payload,
            },
            room=f"user:{uid}",
        )

    # Cluster unknown faces
    cluster_unknown_faces_background(session, event.id)
    print(f"[FaceMatch] Matching completed for media {media_id}")


def cluster_unknown_faces_background(session: Session, event_id: int):
    unknown_matches = session.exec(
        select(FaceMatch).where(
            FaceMatch.event_id == event_id, FaceMatch.matched_user_id == None
        )
    ).all()

    if not unknown_matches:
        return

    embeddings = []
    for match in unknown_matches:
        # Find the media embedding that contains this embedding
        media_embedding = session.exec(
            select(MediaEmbedding).where(MediaEmbedding.media_id == match.media_id)
        ).first()
        if not media_embedding or not media_embedding.embeddings:
            continue

        # Use embedding_index to fetch correct vector
        if match.embedding_index < len(media_embedding.embeddings):
            emb = media_embedding.embeddings[match.embedding_index]
            embeddings.append(emb)

    if not embeddings:
        return

    X = np.array(embeddings)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(X)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(embeddings[idx])

    for label, group in clusters.items():
        cluster = UnknownFaceCluster(
            event_id=event_id, cluster_label=f"cluster_{label}", embeddings=group
        )
        session.add(cluster)

    session.commit()
