from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlmodel import Session
import os
from dotenv import load_dotenv
from db import get_session
from models import FaceEmbedding
from utils.users import get_user_by_id
from utils.face import (
    delete_media_and_file,
    delete_old_face_enrollment,
    generate_face_embedding,
    save_media_file,
)

load_dotenv()
FACE_API_URL = os.environ.get("FACE_API_URL")
FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME")
FACE_MODEL_BACKEND = os.environ.get("FACE_MODEL_BACKEND")
MEDIA_DIR = os.environ.get("MEDIA_DIR")

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/enroll")
async def embed_face(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    user = get_user_by_id(session, user_id)
    if not user:
        raise HTTPException(404, "User not found")

    try:
        # --- Step 1: save new media ---
        media = save_media_file(session, file, user_id)

        # --- Step 2: generate embedding ---
        try:
            file_path = os.path.join(MEDIA_DIR, media.filename)
            embedding = generate_face_embedding(file_path, media.mime_type)
        except Exception as e:
            # if embedding fails → cleanup new media only
            delete_media_and_file(session, media)
            session.commit()
            raise HTTPException(500, f"Face processing failed: {str(e)}")

        # --- Step 3: delete old enrollment only now ---
        delete_old_face_enrollment(session, user_id)

        # --- Step 4: save new embedding ---
        face_embedding = FaceEmbedding(
            user_id=user_id,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_MODEL_BACKEND,
            image_path=file_path,
        )
        face_embedding.set_embedding(embedding)
        session.add(face_embedding)

        # --- Step 5: commit everything together ---
        session.commit()
        session.refresh(face_embedding)

        return {
            "message": "success",
            "embedding_id": face_embedding.id,
            "embedding_length": len(embedding),
            "user_id": user_id,
            "media_id": media.id,
            "image_url": media.url,
        }

    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(500, f"Unexpected error: {str(e)}")
