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
from utils.ws import notify_ws
from fastapi.encoders import jsonable_encoder


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


def generate_embeddings_background(media_id: int, image_url: str):
    with next(get_session()) as session:
        media = session.get(Media, media_id)
        if not media:
            print(f"Media {media_id} not found")
            return

        embedding = MediaEmbedding(
            media_id=media.id, model_name="ArcFace", status="processing"
        )
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

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
                notify_ws(
                    media.uploaded_by_id,
                    {
                        "type": "embedding_failed",
                        "media_id": media.id,
                        "error": data["error"],
                    },
                )
                return

            embeddings = (
                [face["embedding"] for face in data] if isinstance(data, list) else []
            )
            embedding.embeddings = embeddings
            embedding.status = "completed"
            embedding.processed_at = datetime.utcnow().isoformat()
            session.commit()

            # Notify user: embedding done
            notify_ws(
                media.uploaded_by_id,
                {
                    "type": "embedding_completed",
                    "media_id": media.id,
                    "count": len(embeddings),
                },
            )

            # Trigger matching
            match_faces_background(session, media.id, embedding.id)

        except Exception as e:
            embedding.status = "failed"
            embedding.error_message = str(e)
            session.commit()
            notify_ws(
                media.uploaded_by_id,
                {
                    "type": "embedding_failed",
                    "media_id": media.id,
                    "error": str(e),
                },
            )


def match_faces_background(
    session: Session, media_id: int, embedding_id: int, threshold: float = 0.6
):
    """Match detected faces in an uploaded media against user embeddings and notify matched users."""

    # Load media embedding
    media_embedding = session.get(MediaEmbedding, embedding_id)
    if not media_embedding or not media_embedding.embeddings:
        print(f"[FaceMatch] No embeddings found for MediaEmbedding {embedding_id}")
        return

    # Load media + usage + event
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
            MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
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
    notify_ws(
        event.created_by_id,
        {
            "type": "face_matching_completed",
            "media": media_payload,
            "event_id": event.id,
            "matched_count": len(matched_users),
        },
    )

    # Notify each matched user
    for uid in matched_users:
        notify_ws(
            uid,
            {
                "type": "face_match_batch",
                "event_id": event.id,
                "media": media_payload,
            },
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
