from typing import List
import numpy as np
import requests
from fastapi import UploadFile
from sqlmodel import Session, select
import mimetypes
import os
from models import Media, MediaUsage, FaceEmbedding
from dotenv import load_dotenv

load_dotenv()

FACE_API_URL = os.environ.get("FACE_API_URL")
MEDIA_DIR = os.environ.get("MEDIA_DIR")


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate Euclidean distance between two embeddings"""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.linalg.norm(vec1 - vec2)


def save_media_file(session: Session, file: UploadFile, user_id: int) -> Media:
    file_ext = os.path.splitext(file.filename)[1].lower()
    safe_name = f"face_enrollment_{user_id}_{os.urandom(8).hex()}{file_ext}"
    file_path = os.path.join(MEDIA_DIR, safe_name)

    content = file.file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or file.content_type or "application/octet-stream"

    media = Media(
        url=f"/media/{safe_name}",
        filename=safe_name,
        original_filename=file.filename,
        file_size=os.path.getsize(file_path),
        mime_type=mime_type,
        uploaded_by_id=user_id,
    )
    session.add(media)
    session.flush()  # assign media.id

    usage = MediaUsage(
        owner_id=user_id,
        owner_type="user",
        usage_type="profile_picture",
        media_type="image",
        media_id=media.id,
    )
    session.add(usage)
    session.flush()
    return media


def generate_face_embedding(file_path: str, mime_type: str):
    with open(file_path, "rb") as f:
        files = {"file": ("image.jpg", f, mime_type)}
        resp = requests.post(f"{FACE_API_URL}/embed", files=files)
        resp.raise_for_status()
        result = resp.json()[0]

        embedding = result.get("embedding")
        if not embedding:
            raise ValueError(f"Face API did not return embedding: {result}")

        return embedding


def delete_media_and_file(session: Session, media: Media):
    """Delete a media row, its usage, and file from disk."""
    if not media:
        return

    # Delete file
    file_path = os.path.join(MEDIA_DIR, media.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete associated usages
    usages = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media.id)
    ).all()
    for usage in usages:
        session.delete(usage)

    # Delete media
    session.delete(media)
    session.flush()


def delete_old_face_enrollment(session: Session, user_id: int):
    """Delete old face embedding + its media/usage."""
    old_embedding = session.exec(
        select(FaceEmbedding).where(FaceEmbedding.user_id == user_id)
    ).first()
    if old_embedding:
        session.delete(old_embedding)

    old_usage = session.exec(
        select(MediaUsage).where(
            MediaUsage.owner_id == user_id,
            MediaUsage.owner_type == "user",
            MediaUsage.usage_type == "face_enrollment",
        )
    ).first()
    if old_usage:
        old_media = session.get(Media, old_usage.media_id)
        delete_media_and_file(session, old_media)
