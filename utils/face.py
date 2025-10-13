import os
import numpy as np
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlmodel import Session, select
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from models.media import Media, MediaUsage
from models import (
    FaceEmbedding,
    FaceCluster,
    Event,
    User,
    ContentOwnerType,
    MediaUsageType,
)
from tasks.notifications import send_notification


load_dotenv()

FACE_API_URL = os.getenv("FACE_API_URL")
CLUSTER_SIMILARITY_THRESHOLD = 0.68
USER_MATCH_SIMILARITY = 0.72


class NetworkError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(NetworkError),
    reraise=True,
)
def safe_request(method: str, url: str, **kwargs):
    try:
        r = requests.request(method, url, timeout=60, **kwargs)
        r.raise_for_status()
        return r
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
        raise NetworkError(str(e))


def validate_embedding(embedding):
    """
    Validate that an embedding contains valid float values (no NaN or Inf).
    Returns True if valid, False otherwise.
    """
    if embedding is None or len(embedding) == 0:
        return False

    arr = np.array(embedding, dtype=float)

    # Check for NaN or Inf values
    if np.isnan(arr).any() or np.isinf(arr).any():
        return False

    return True


def create_cluster_safely(session: Session, centroid_array):
    """
    Create a FaceCluster with validation.
    Returns the created cluster or None if validation fails.
    """
    if not validate_embedding(centroid_array):
        print(
            f"[cluster] Invalid centroid detected (NaN/Inf), skipping cluster creation"
        )
        return None

    new_c = FaceCluster(
        centroid=centroid_array.tolist(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(new_c)
    session.commit()
    session.refresh(new_c)
    return new_c


def generate_embeddings_background(session: Session, media_id: int, image_url: str):
    """
    Fetch image, call FACE_API_URL/embed, create FaceEmbedding rows for each detected face,
    then cluster the new embeddings and try matching clusters to users.
    """
    media = session.get(Media, media_id)
    if not media:
        print(f"[embed] media {media_id} not found")
        return

    try:
        # download image
        res = safe_request("GET", image_url)

        # call embedding API
        resp = safe_request(
            "POST",
            f"{FACE_API_URL}/embed",
            files={"file": ("image.jpg", res.content, "image/jpeg")},
        )
        data = resp.json()

        # API returns list of face dicts (embedding + facial_area)
        if isinstance(data, dict) and data.get("error"):
            send_notification.delay(
                user_id=media.uploaded_by_id,
                event="embedding_failed",
                data={"media_id": media.id, "error": data.get("error")},
            )
            return

        faces = data if isinstance(data, list) else []
        created = 0
        for idx, face in enumerate(faces):
            emb = face.get("embedding")
            facial_area = face.get("facial_area")
            if not emb:
                continue

            # Validate embedding before saving
            if not validate_embedding(emb):
                print(
                    f"[embed] Invalid embedding for face {idx} in media {media_id}, skipping"
                )
                continue

            fe = FaceEmbedding(
                media_id=media.id,
                embedding=emb,
                facial_area=facial_area,
                cluster_id=None,
                created_at=datetime.now(timezone.utc),
            )
            session.add(fe)
            created += 1

        # update media face_count and commit
        media.face_count = created
        session.commit()

        # Fire an embedding-completed notification to uploader
        send_notification.delay(
            user_id=media.uploaded_by_id,
            event="embedding_completed",
            data={"media_id": media.id, "count": created},
        )

        # Next steps: cluster the new embeddings and try to match to users
        cluster_embeddings_for_media(session, media_id)
        match_clusters_to_users(session, media_id)

        print(f"[embed] processed {created} faces for media {media_id}")

    except Exception as e:
        print(f"[embed] error for media {media_id}: {e}")
        send_notification.delay(
            user_id=media.uploaded_by_id,
            event="embedding_failed",
            data={"media_id": media.id, "error": str(e)},
        )


def cluster_embeddings_for_media(
    session: Session, media_id: int, threshold: float = CLUSTER_SIMILARITY_THRESHOLD
):
    """
    For each FaceEmbedding of this media that has no cluster,
    either assign it to a nearest FaceCluster (if similarity >= threshold)
    or create a new FaceCluster for it.
    """
    face_embeddings = session.exec(
        select(FaceEmbedding).where(
            FaceEmbedding.media_id == media_id, FaceEmbedding.cluster_id == None
        )
    ).all()

    if not face_embeddings:
        return

    # load all existing cluster centroids (global)
    clusters = session.exec(select(FaceCluster)).all()
    cluster_centroids = (
        [np.array(c.centroid, dtype=float) for c in clusters] if clusters else []
    )

    updated_cluster_ids = set()

    for fe in face_embeddings:
        # Validate embedding before processing
        if not validate_embedding(fe.embedding):
            print(f"[cluster] Invalid embedding for FaceEmbedding {fe.id}, skipping")
            continue

        vec = np.array(fe.embedding, dtype=float)

        # if no clusters exist -> create new
        if not cluster_centroids:
            new_c = create_cluster_safely(session, vec)
            if new_c:
                fe.cluster_id = new_c.id
                session.add(fe)
                session.commit()
                updated_cluster_ids.add(new_c.id)
                # refresh cluster lists
                clusters = session.exec(select(FaceCluster)).all()
                cluster_centroids = [
                    np.array(c.centroid, dtype=float) for c in clusters
                ]
            continue

        # compute cosine similarity to each cluster centroid
        try:
            sims = cosine_similarity([vec], cluster_centroids)[0]
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
        except Exception as e:
            print(
                f"[cluster] Error computing similarity for FaceEmbedding {fe.id}: {e}"
            )
            continue

        if best_score >= threshold:
            cluster = clusters[best_idx]
            fe.cluster_id = cluster.id
            session.add(fe)
            session.commit()
            updated_cluster_ids.add(cluster.id)
            update_cluster_centroid(session, cluster)
        else:
            # create new cluster
            new_c = create_cluster_safely(session, vec)
            if new_c:
                fe.cluster_id = new_c.id
                session.add(fe)
                session.commit()
                updated_cluster_ids.add(new_c.id)
                # refresh cluster lists for subsequent iterations
                clusters = session.exec(select(FaceCluster)).all()
                cluster_centroids = [
                    np.array(c.centroid, dtype=float) for c in clusters
                ]

    # ensure centroids refreshed for all updated clusters
    for cid in updated_cluster_ids:
        cluster = session.get(FaceCluster, cid)
        update_cluster_centroid(session, cluster)


def update_cluster_centroid(session: Session, cluster: FaceCluster):
    """
    Recalculate centroid as mean of all embeddings currently assigned to this cluster.
    """
    if cluster is None:
        return

    face_rows = session.exec(
        select(FaceEmbedding).where(FaceEmbedding.cluster_id == cluster.id)
    ).all()

    if not face_rows:
        return

    # Filter out invalid embeddings
    valid_embeddings = [
        np.array(f.embedding, dtype=float)
        for f in face_rows
        if validate_embedding(f.embedding)
    ]

    if not valid_embeddings:
        print(f"[cluster] No valid embeddings for cluster {cluster.id}")
        return

    mat = np.array(valid_embeddings)
    centroid = mat.mean(axis=0)

    # Validate the computed centroid
    if not validate_embedding(centroid):
        print(
            f"[cluster] Computed centroid for cluster {cluster.id} is invalid, skipping update"
        )
        return

    cluster.centroid = centroid.tolist()
    cluster.updated_at = datetime.now(timezone.utc)
    session.add(cluster)
    session.commit()


def match_clusters_to_users(
    session: Session, media_id: int, user_threshold: float = USER_MATCH_SIMILARITY
):
    """
    For clusters that contain embeddings from this media, try to match the cluster centroid
    to users who have profile picture embeddings. If match found, set cluster.user_id and notify.
    """
    usage = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media_id)
    ).first()
    event = None
    if usage and usage.owner_type == ContentOwnerType.EVENT:
        event = session.get(Event, usage.owner_id)

    # get clusters touched by this media
    touched_cluster_ids = session.exec(
        select(FaceEmbedding.cluster_id).where(
            FaceEmbedding.media_id == media_id, FaceEmbedding.cluster_id != None
        )
    ).all()
    touched_cluster_ids = list({cid for cid in touched_cluster_ids if cid is not None})
    if not touched_cluster_ids:
        return

    # gather user profile embeddings
    user_profile_rows = session.exec(
        select(User, FaceEmbedding)
        .join(MediaUsage, MediaUsage.owner_id == User.id)
        .join(Media, Media.id == MediaUsage.media_id)
        .join(FaceEmbedding, FaceEmbedding.media_id == Media.id)
        .where(
            MediaUsage.owner_type == ContentOwnerType.USER,
            MediaUsage.usage_type.in_(
                [
                    MediaUsageType.PROFILE_PICTURE,
                    MediaUsageType.PROFILE_PICTURE_ARCHIVED,
                ]
            ),
        )
    ).all()

    # build map user_id -> embedding vector
    user_embeddings = {}
    for user, face_emb in user_profile_rows:
        if face_emb and face_emb.embedding and validate_embedding(face_emb.embedding):
            user_embeddings[user.id] = np.array(face_emb.embedding, dtype=float)

    # for each touched cluster, compute similarity to profile embeddings
    for cid in touched_cluster_ids:
        cluster = session.get(FaceCluster, cid)
        if not cluster or not validate_embedding(cluster.centroid):
            continue

        centroid = np.array(cluster.centroid, dtype=float)

        best_user = None
        best_sim = -1.0
        for uid, uvec in user_embeddings.items():
            try:
                sim = float(cosine_similarity([centroid], [uvec])[0][0])
                if sim > best_sim:
                    best_sim = sim
                    best_user = uid
            except Exception as e:
                print(
                    f"[match] Error computing similarity for cluster {cid} and user {uid}: {e}"
                )
                continue

        # assign if above threshold and cluster not already linked to same user
        if best_user and best_sim >= user_threshold:
            if cluster.user_id != best_user:
                cluster.user_id = best_user
                session.add(cluster)
                session.commit()

                # notify matched user
                send_notification.delay(
                    user_id=best_user,
                    event="cluster_matched_to_user",
                    data={
                        "cluster_id": cluster.id,
                        "media_id": media_id,
                        "similarity": best_sim,
                        "event_id": event.id if event else None,
                    },
                )

                # notify event owner
                if event and event.created_by_id:
                    send_notification.delay(
                        user_id=event.created_by_id,
                        event="cluster_user_identified",
                        data={
                            "cluster_id": cluster.id,
                            "user_id": best_user,
                            "similarity": best_sim,
                            "media_id": media_id,
                        },
                    )


def cluster_unclustered_faces_in_event(
    session: Session, event_id: int, eps: float = 0.5, min_samples: int = 2
):
    """
    For any FaceEmbedding in an event with cluster_id == NULL, run DBSCAN to group them
    and create new FaceCluster rows for each found group.
    """
    rows = session.exec(
        select(FaceEmbedding, Media)
        .join(Media, Media.id == FaceEmbedding.media_id)
        .join(MediaUsage, MediaUsage.media_id == Media.id)
        .where(
            MediaUsage.owner_type == ContentOwnerType.EVENT,
            MediaUsage.owner_id == event_id,
        )
    ).all()

    # filter only unclustered embeddings with valid data
    unclustered = []
    mapping = []
    for fe, media in rows:
        if fe.cluster_id is None and validate_embedding(fe.embedding):
            unclustered.append(np.array(fe.embedding, dtype=float))
            mapping.append(fe)

    if not unclustered:
        return

    X = np.array(unclustered)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
    labels = db.labels_

    # group by label
    groups = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        groups.setdefault(label, []).append(mapping[i])

    for label, group in groups.items():
        # create FaceCluster using mean of group
        valid_embeddings = [
            np.array(g.embedding, dtype=float)
            for g in group
            if validate_embedding(g.embedding)
        ]

        if not valid_embeddings:
            continue

        mat = np.array(valid_embeddings)
        centroid = mat.mean(axis=0)

        # Validate centroid before creating cluster
        if not validate_embedding(centroid):
            print(
                f"[dbscan] Invalid centroid for group {label} in event {event_id}, skipping"
            )
            continue

        cluster = FaceCluster(
            centroid=centroid.tolist(),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(cluster)
        session.commit()

        # assign face embeddings in group to this cluster
        for fe in group:
            fe.cluster_id = cluster.id
            session.add(fe)
        session.commit()

        # notify event owner about a new unknown cluster
        send_notification.delay(
            user_id=event_id,
            event="unknown_cluster_created",
            data={
                "event_id": event_id,
                "cluster_id": cluster.id,
                "count": len(group),
            },
        )
