from fastapi import APIRouter, UploadFile, File
import requests
import os
from dotenv import load_dotenv

load_dotenv()
FACE_API_URL = os.environ.get("FACE_API_URL")

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/embed")
async def embed_face(file: UploadFile = File(...)):
    try:
        # Pass the file directly as a file-like object
        files = {"file": (file.filename, file.file, file.content_type)}
        response = requests.post(f"{FACE_API_URL}/embed", files=files)
        result = response.json()
        return result
    except Exception as e:
        return {"error": str(e), "from": "main server"}
