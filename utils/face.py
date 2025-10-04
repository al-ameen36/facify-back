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
)
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


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

        media = session.get(Media, media_id)
        if not media:
            return

        embedding = MediaEmbedding(
            media_id=media.id, model_name="ArcFace", status="processing"
        )
        session.add(embedding)
        session.commit()
        session.refresh(embedding)

        try:
            # 1️⃣ Fetch image from ImageKit (with retries)
            res = safe_request("GET", image_url)

            # 2️⃣ Send to DeepFace server (with retries)
            response = safe_request(
                "POST",
                f"{FACE_API_URL}/embed",
                files={"file": ("image.jpg", res.content, "image/jpeg")},
            )

            data = response.json()

            # 3️⃣ Process response
            if "error" in data:
                embedding.status = "failed"
                embedding.error_message = data["error"]
            else:
                embeddings = (
                    [face["embedding"] for face in data]
                    if isinstance(data, list)
                    else []
                )
                embedding.embeddings = embeddings
                embedding.status = "completed"
                embedding.processed_at = datetime.utcnow().isoformat()

            session.add(embedding)
            session.commit()

            # 4️⃣ Trigger matching if successful
            if embedding.status == "completed":
                match_faces_background(session, media.id, embedding.id)

        except Exception as e:
            embedding.status = "failed"
            embedding.error_message = str(e)
            session.add(embedding)
            session.commit()


def match_faces_background(
    session: Session, media_id: int, embedding_id: int, threshold: float = 0.6
):
    """Match detected faces in an uploaded media against all users with known embeddings."""
    from scipy.spatial.distance import cosine
    from models import ContentOwnerType, MediaUsageType

    # 1️⃣ Load media embedding
    media_embedding = session.get(MediaEmbedding, embedding_id)
    if not media_embedding or not media_embedding.embeddings:
        print(f"[FaceMatch] No embeddings found for MediaEmbedding {embedding_id}")
        return

    # 2️⃣ Identify event context
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

    # 3️⃣ Get approved participants
    participant_ids = [
        ep.user_id
        for ep in session.exec(
            select(EventParticipant.user_id).where(
                EventParticipant.event_id == event.id,
                EventParticipant.status == "approved",
            )
        ).all()
    ]

    # 4️⃣ Get all users with completed embeddings (profile pictures)
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

    # 5️⃣ Compare each detected face embedding against all user embeddings
    for idx, emb in enumerate(media_embedding.embeddings or []):
        best_user = None
        best_distance = 1.0

        for user, user_embedding in users_with_embeddings:
            if not user_embedding.embeddings:
                continue

            # Compare with the first (or average) embedding of the user
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

        print(
            f"[FaceMatch] Face #{idx}: "
            f"{'Matched ' + best_user.username if is_match else 'Unknown face'} "
            f"(distance={best_distance:.4f})"
        )

    session.commit()

    # 6️⃣ Cluster unknowns for further analysis
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
