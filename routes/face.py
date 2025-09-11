from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlmodel import Session
import requests
import os
from dotenv import load_dotenv
from db import get_session
from models import FaceEmbedding
from utils.users import get_user_by_id

load_dotenv()
FACE_API_URL = os.environ.get("FACE_API_URL")
FACE_MODEL_NAME = os.environ.get("FACE_MODEL_NAME")
FACE_MODEL_BACKEND = os.environ.get("FACE_MODEL_BACKEND")

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/embed")
async def embed_face(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    temp_path = None
    try:
        user = get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Pass the file directly as a file-like object
        files = {"file": (file.filename, file.file, file.content_type)}
        response = requests.post(f"{FACE_API_URL}/embed", files=files)
        result = response.json()
        embedding = result[0]["embedding"]
        confidence_score = result[0]["confidence_score"]

        face_embedding = FaceEmbedding(
            user_id=user_id,
            model_name=FACE_MODEL_NAME,
            detector_backend=FACE_MODEL_BACKEND,
            image_path=temp_path,
            confidence_score=confidence_score,
        )
        face_embedding.set_embedding(embedding)

        session.add(face_embedding)
        session.commit()
        session.refresh(face_embedding)

        return {
            "message": "success",
            "embedding_id": face_embedding.id,
            "embedding_length": len(embedding),
            "model_name": FACE_MODEL_NAME,
            "user_id": user_id,
        }

    except Exception as e:
        session.rollback()
        return {"error": str(e), "source": "main server"}
