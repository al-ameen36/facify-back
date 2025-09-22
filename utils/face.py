from typing import List
import numpy as np
import requests
from fastapi import UploadFile
from sqlmodel import Session, select
import mimetypes
import tempfile
import os
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from models import Media, MediaUsage, MediaEmbedding, User
from utils.drive import get_drive_service
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
# Drive helpers
# ------------------------------
def download_drive_file(drive_service, file_id, local_path):
    request = drive_service.files().get_media(fileId=file_id)
    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    return local_path


def get_or_create_app_folders(drive_service, app_name="MyApp"):
    """Ensure app folders exist: [app_name]/images, videos, docs"""
    # Step 1: root folder
    q = f"name='{app_name}' and mimeType='application/vnd.google-apps.folder' and 'root' in parents and trashed=false"
    results = drive_service.files().list(q=q, fields="files(id)").execute()
    if results["files"]:
        root_id = results["files"][0]["id"]
    else:
        root_id = (
            drive_service.files()
            .create(
                body={
                    "name": app_name,
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": ["root"],
                },
                fields="id",
            )
            .execute()["id"]
        )

    # Step 2: subfolders
    folders = {}
    for name in ["images", "videos", "docs"]:
        q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{root_id}' in parents and trashed=false"
        res = drive_service.files().list(q=q, fields="files(id)").execute()
        if res["files"]:
            folders[name] = res["files"][0]["id"]
        else:
            folders[name] = (
                drive_service.files()
                .create(
                    body={
                        "name": name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [root_id],
                    },
                    fields="id",
                )
                .execute()["id"]
            )
    return folders


def determine_folder_id(drive_service, mime_type: str, folders: dict):
    if mime_type.startswith("image/"):
        return folders["images"]
    elif mime_type.startswith("video/"):
        return folders["videos"]
    else:
        return folders["docs"]


# ------------------------------
# Media ops
# ------------------------------
def save_media_file(
    session: Session,
    file: UploadFile,
    user: User,
    owner_id: int,
    owner_type: str,
    usage_type: str,
) -> Media:
    """Upload file to the user's Drive and save metadata in DB."""

    drive_service = get_drive_service(user)

    # create folders if missing
    folders = get_or_create_app_folders(drive_service, app_name=APP_NAME)

    # choose folder based on mime
    mime_type = (
        file.content_type
        or mimetypes.guess_type(file.filename)[0]
        or "application/octet-stream"
    )
    folder_id = determine_folder_id(drive_service, mime_type, folders)

    # upload to Drive
    media_body = MediaIoBaseUpload(file.file, mimetype=mime_type, resumable=True)
    uploaded = (
        drive_service.files()
        .create(
            body={"name": file.filename, "parents": [folder_id]},
            media_body=media_body,
            fields="id, webViewLink, mimeType, size",
        )
        .execute()
    )

    file_id = uploaded["id"]
    file_url = uploaded["webViewLink"]
    file_size = int(uploaded.get("size", 0))

    # save Media
    media = Media(
        external_url=file_url,
        filename=file.filename,
        original_filename=file.filename,
        file_size=file_size,
        mime_type=mime_type,
        uploaded_by_id=user.id,
        external_id=file_id,
    )
    session.add(media)
    session.flush()

    # save MediaUsage (dynamic now)
    usage = MediaUsage(
        owner_id=owner_id,
        owner_type=owner_type,
        usage_type=usage_type,
        media_type=(
            "image" if mime_type.startswith("image/") else mime_type.split("/")[0]
        ),
        media_id=media.id,
    )
    session.add(usage)
    session.flush()

    return media


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


def delete_media_and_file(session: Session, media: Media, user: User):
    """Delete a media row + usage + Drive file."""
    if not media:
        return

    drive_service = get_drive_service(user)

    # Delete from Drive
    if media.external_id:
        try:
            drive_service.files().delete(fileId=media.external_id).execute()
        except Exception:
            pass  # file may already be gone

    # Delete usages
    usages = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media.id)
    ).all()
    for usage in usages:
        session.delete(usage)

    session.delete(media)
    session.flush()


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
