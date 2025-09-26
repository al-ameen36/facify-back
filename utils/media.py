import os
import shutil
import tempfile
from typing import List
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from dotenv import load_dotenv
from sqlmodel import Session, select
from fastapi import UploadFile
from models import User, Media, MediaUsage, MediaEmbedding
import mimetypes

load_dotenv()

PRIVATE_KEY = os.environ.get("IK_PRIVATE_KEY")
PUBLIC_KEY = os.environ.get("IK_PUBLIC_KEY")
URL = os.environ.get("IK_URL")

imagekit = ImageKit(private_key=PRIVATE_KEY, public_key=PUBLIC_KEY, url_endpoint=URL)


def get_mime_type(file_path: str) -> str:
    """
    Returns the MIME type of a file based on its extension.
    Falls back to 'application/octet-stream' if unknown.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def upload_file(
    file: UploadFile,
    user: User,
    owner_id: int,
    owner_type: str,
    usage_type: str,
    use_unique_file_name: bool = True,
):
    # Hybrid approach: choose strategy based on file size
    MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB threshold
    
    # Get file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size <= MAX_MEMORY_SIZE:
        # Small files: use memory approach (faster)
        return _upload_via_memory(file, user, owner_id, owner_type, usage_type, use_unique_file_name)
    else:
        # Large files: use temp file approach (memory efficient)
        return _upload_via_temp_file(file, user, owner_id, owner_type, usage_type, use_unique_file_name)


def _upload_via_memory(file, user, owner_id, owner_type, usage_type, use_unique_file_name):
    """Upload small files via memory (BytesIO)"""
    import io
    
    file.file.seek(0)
    file_content = file.file.read()
    file.file.seek(0)
    
    file_stream = io.BytesIO(file_content)
    
    upload = imagekit.upload_file(
        file=file_stream,
        file_name=file.filename or "uploaded_file",
        options=UploadFileRequestOptions(
            use_unique_file_name=use_unique_file_name,
            tags=[
                owner_type,
                f"owner-{owner_id}",
                f"creator-{user.id}",
                f"usage-{usage_type}",
            ],
        ),
    )
    
    return _process_upload_response(upload, file)


def _upload_via_temp_file(file, user, owner_id, owner_type, usage_type, use_unique_file_name):
    """Upload large files via temp file (memory efficient)"""
    file_path = save_upload_file_to_temp(file)

    try:
        with open(file_path, "rb") as f:
            upload = imagekit.upload_file(
                file=f,
                file_name=file.filename or "uploaded_file",
                options=UploadFileRequestOptions(
                    use_unique_file_name=use_unique_file_name,
                    tags=[
                        owner_type,
                        f"owner-{owner_id}",
                        f"creator-{user.id}",
                        f"usage-{usage_type}",
                    ],
                ),
            )
    finally:
        # Clean up temp file
        try:
            os.remove(file_path)
        except OSError:
            pass
    
    return _process_upload_response(upload, file)


def _process_upload_response(upload, file):
    """Process ImageKit upload response into Media object"""
    raw = upload.response_metadata.raw

    media_data = {
        "external_id": raw.get("fileId"),
        "external_url": raw.get("url"),
        "filename": raw.get("name", "unknown"),
        "original_filename": file.filename,
        "file_size": raw.get("size"),
        "mime_type": get_mime_type(file.filename or ""),
        "duration": raw.get("duration", 0),
    }

    return Media(**media_data)




def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    """
    Save a FastAPI UploadFile to a temporary file and return its path.
    """
    suffix = (
        upload_file.filename.split(".")[-1]
        if upload_file.filename and "." in upload_file.filename
        else ""
    )
    suffix = f".{suffix}" if suffix else ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, tmp)
        temp_file_path = tmp.name

    return temp_file_path


def save_file_to_db(
    session: Session,
    media: Media,
    owner_id: int,
    owner_type: str,
    usage_type: str,
    user_id: int = None,
    embeddings: List[List[float]] = None,
):
    session.add(media)
    session.flush()

    usage = MediaUsage(
        owner_id=owner_id,
        owner_type=owner_type,
        usage_type=usage_type,
        media_type=media.mime_type,
        media_id=media.id,
    )
    session.add(usage)
    session.flush()

    media_embedding = MediaEmbedding(
        media_id=media.id,
        model_name="Facenet",
        user_id=(user_id if usage_type == "face_enrollment" else None),
        embeddings=embeddings,
        status="completed" if embeddings else "pending",
    )
    session.add(media_embedding)
    session.commit()

    return media


def delete_media_and_file(session: Session, media: Media, user: User):
    """Delete a media row + embedding + usages + ImageKit file, only if owned by the user."""
    if not media:
        return

    # Get usages for this media
    usages = session.exec(
        select(MediaUsage).where(MediaUsage.media_id == media.id)
    ).all()

    # Ownership check
    if not all(usage.owner_id == user.id for usage in usages):
        raise PermissionError("You are not allowed to delete this media")

    # Delete from ImageKit
    if media.external_id:
        try:
            imagekit.delete_file(media.external_id)
        except Exception:
            pass

    # Delete embedding
    embedding = session.exec(
        select(MediaEmbedding).where(MediaEmbedding.media_id == media.id)
    ).first()
    if embedding:
        session.delete(embedding)

    # Delete usages
    for usage in usages:
        session.delete(usage)

    # Delete the media itself
    session.delete(media)
    session.flush()
