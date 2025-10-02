from googleapiclient.http import MediaFileUpload
import google.oauth2.credentials
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import os
from models import User

load_dotenv()


CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
SCOPES = [os.environ.get("SCOPES")]

APP_FOLDER_NAME = os.environ.get("APP_NAME")


def get_drive_service(user: User):
    """Build Google Drive service from stored tokens."""
    creds = Credentials(
        token=user.drive_access_token,
        refresh_token=user.drive_refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    )
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, name, parent_id="root"):
    """Return the folder ID. If it doesn’t exist, create it."""
    try:
        query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get("files", [])

        if items:
            return items[0]["id"]

        # Create folder
        file_metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = service.files().create(body=file_metadata, fields="id").execute()
        return folder["id"]

    except HttpError as e:
        print(f"An error occurred: {e}")
        raise


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
