from googleapiclient.http import MediaFileUpload
import google.oauth2.credentials
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from models import User

load_dotenv()


CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def upload_to_drive(user: User, file_path: str, filename: str):
    creds = google.oauth2.credentials.Credentials(
        token=user.drive_access_token,
        refresh_token=user.drive_refresh_token,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
    )
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": filename}
    media = MediaFileUpload(file_path, resumable=True)

    uploaded = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id, webViewLink")
        .execute()
    )

    return uploaded
