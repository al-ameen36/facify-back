from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from models import User
import os
from dotenv import load_dotenv

load_dotenv()


def get_drive_service(user: User):
    """Build Google Drive service from stored tokens."""
    creds = Credentials(
        token=user.google_access_token,
        refresh_token=user.google_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    )
    return build("drive", "v3", credentials=creds)
