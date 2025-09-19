from datetime import datetime
from models.media import MediaUsage
from sqlmodel import SQLModel, Field, Relationship, select
from typing import List, Optional
from models.core import AppBaseModel, ContentOwnerType, MediaUsageType
import random
import string
import io
import base64
from googleapiclient.http import MediaIoBaseDownload


def generate_event_secret() -> str:
    """Generate a secret code like Google Meet style: abc-defg-hij"""
    parts = []
    lengths = [3, 4, 3]  # like Google Meet
    for length in lengths:
        part = "".join(random.choice(string.ascii_lowercase) for _ in range(length))
        parts.append(part)
    return "-".join(parts)


# Types
class EventBase(SQLModel):
    name: str = Field(index=True, unique=True)
    location: str
    description: str
    start_time: datetime
    end_time: datetime
    privacy: str
    allow_contributions: Optional[bool] = True
    auto_approve_uploads: Optional[bool] = True
    privacy: str


class EventRead(EventBase):
    id: int
    created_by_id: int
    updated_at: datetime
    created_at: datetime
    cover_photo: Optional[str] = None
    secret: Optional[str] = None


class EventCreate(EventBase):
    pass


class EventCreateDB(EventBase):
    created_by_id: int


class ParticipantRead(SQLModel):
    id: int
    full_name: str
    username: str
    photo: Optional[str] = None
    email: str
    status: str
    created_at: datetime


class JoinEventRequest(SQLModel):
    secret: str


# Models
class EventParticipant(AppBaseModel, table=True):
    """Junction table for many-to-many relationship between events and users"""

    event_id: int = Field(foreign_key="event.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    status: str = Field(default="pending")


class Event(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    location: str
    start_time: datetime
    end_time: datetime
    privacy: str
    secret: Optional[str] = Field(default_factory=generate_event_secret)
    allow_contributions: Optional[bool] = True
    auto_approve_uploads: Optional[bool] = True
    privacy: Optional[str] = "private"

    # Relationships
    created_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_by: Optional["User"] = Relationship(back_populates="events")
    participants: List["User"] = Relationship(back_populates="joined_events")

    def get_cover_photo_media(self, session) -> Optional["Media"]:
        """
        Returns the Media object for the cover photo.
        Returns None if no cover photo exists.
        """
        usage = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.EVENT,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.COVER_PHOTO,
            )
        ).first()

        if not usage or not usage.media:
            return None

        return usage.media

    def get_cover_photo_base64(
        self, session, user: "User", drive_service=None
    ) -> Optional[str]:
        """
        Returns the cover photo as a base64 string ready for frontend <img src="...">.
        Returns None if no cover photo exists or fetching fails.
        """
        media = self.get_cover_photo_media(session)
        if not media or not media.external_id:
            return None

        try:
            if not drive_service:
                from utils.drive import get_drive_service
                drive_service = get_drive_service(user)

            request = drive_service.files().get_media(fileId=media.external_id)
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            file_stream.seek(0)
            encoded = base64.b64encode(file_stream.read()).decode()
            return f"data:{media.mime_type};base64,{encoded}"
        except Exception:
            # Fail gracefully
            return None
