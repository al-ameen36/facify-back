import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from db import get_session
from models import User, UserRead
from utils.users import get_current_user
from utils.drive import setup_user_drive_structure
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

router = APIRouter(prefix="/drive", tags=["drive"])

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
SCOPES = [os.environ.get("SCOPES")]


@router.get("/connect")
async def connect_drive(current_user: User = Depends(get_current_user)):
    from urllib.parse import urlencode

    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }

    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return {"auth_url": url}


class CodeSchema(BaseModel):
    code: str


@router.post("/callback")
async def drive_callback(
    code: CodeSchema,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code.code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    resp = requests.post(token_url, data=data)
    tokens = resp.json()

    if "access_token" not in tokens:
        raise HTTPException(status_code=400, detail="Failed to fetch tokens")

    # Save tokens in DB (add drive_access_token, drive_refresh_token fields to User model)
    current_user.drive_access_token = tokens["access_token"]
    current_user.drive_refresh_token = tokens.get("refresh_token")
    current_user.is_drive_connected = True
    session.add(current_user)
    session.commit()
    session.refresh(current_user)

    setup_user_drive_structure(current_user)

    return {
        "message": "Google Drive connected successfully",
        "user": UserRead.model_validate(current_user),
    }
