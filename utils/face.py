from typing import List
import numpy as np
import requests
from fastapi import UploadFile
from sqlmodel import Session, select
import os
from models import Media, MediaUsage, MediaEmbedding, User
from dotenv import load_dotenv

load_dotenv()

FACE_API_URL = os.environ.get("FACE_API_URL")
APP_NAME = os.environ.get("APP_NAME")


# ------------------------------
# Embedding math
# ------------------------------
def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    dot_product = np.dot(vec1, vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
    return np.linalg.norm(np.array(embedding1) - np.array(embedding2))


# ------------------------------
# Media ops
# ------------------------------


def generate_face_embeddings(file: UploadFile) -> list[list[float]]:
    """
    Call FACE API.
    Returns a list of embeddings (one per detected face).
    """
    # Read file contents
    file_bytes = file.file.read()

    resp = requests.post(
        f"{FACE_API_URL}/embed",
        files={"file": (file.filename, file_bytes, file.content_type)},
    )
    resp.raise_for_status()
    results = resp.json()

    embeddings = []
    if isinstance(results, list):
        for r in results:
            if "embedding" in r:
                embeddings.append(r["embedding"])

    if not embeddings:
        raise ValueError(f"Face API did not return embeddings: {results}")

    file.file.seek(0)
    return embeddings


def delete_old_face_enrollment(session: Session, user: User):
    """Delete old face embedding + its Drive media/usage."""
    old_embedding = session.exec(
        select(MediaEmbedding).where(MediaEmbedding.user_id == user.id)
    ).first()
    if old_embedding:
        session.delete(old_embedding)

    old_usage = session.exec(
        select(MediaUsage).where(
            MediaUsage.owner_id == user.id,
            MediaUsage.owner_type == "user",
            MediaUsage.usage_type == "face_enrollment",
        )
    ).first()
    if old_usage:
        old_media = session.get(Media, old_usage.media_id)
        delete_media_and_file(session, old_media, user)
