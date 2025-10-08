from datetime import datetime
from typing import List, Optional
from sqlmodel import (
    SQLModel,
    Field,
    Relationship,
    select,
    Column,
    String,
    ForeignKey,
    Integer,
)
from models.core import AppBaseModel, ContentOwnerType, MediaUsageType
from models.media import MediaUsage, MediaRead
import random
import string


def generate_event_secret() -> str:
    """Generate a secret code like Google Meet style: abc-defg-hij"""
    parts = []
    lengths = [3, 4, 3]  # like Google Meet
    for length in lengths:
        part = "".join(random.choice(string.ascii_lowercase) for _ in range(length))
        parts.append(part)
    return "-".join(parts)


# --- Pydantic/SQLModel Schemas ---
class EventBase(SQLModel):
    name: str = Field(index=True, unique=True)
    location: str
    description: str
    start_time: datetime
    end_time: datetime
    privacy: str
    allow_contributions: bool = True
    auto_approve_uploads: bool = True


class EventRead(EventBase):
    id: int
    created_by_id: int
    updated_at: datetime
    created_at: datetime
    cover_photo: Optional[MediaRead] = None
    secret: Optional[str] = None


class EventCreate(EventBase):
    pass


class EventCreateDB(EventBase):
    created_by_id: int


class ParticipantRead(SQLModel):
    id: int
    full_name: str
    username: str
    profile_picture: Optional[MediaRead] = None
    email: str
    status: str
    created_at: datetime


class JoinEventRequest(SQLModel):
    secret: str


# --- Database Models ---
class EventParticipant(AppBaseModel, table=True):
    """Junction table for many-to-many relationship between events and users"""

    event_id: int = Field(foreign_key="event.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    status: str = Field(default="pending", sa_column=Column(String, index=True))


class Event(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str
    location: str
    start_time: datetime
    end_time: datetime
    privacy: str = "private"
    secret: str = Field(default_factory=generate_event_secret)
    allow_contributions: bool = True
    auto_approve_uploads: bool = True
    # Relationships
    created_by_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", name="fk_event_created_by_id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    created_by: Optional["User"] = Relationship(back_populates="events")
    participants: List["User"] = Relationship(
        back_populates="joined_events",
        link_model=EventParticipant,
    )

    def get_cover_photo_media(self, session) -> Optional[MediaRead]:
        """
        Returns the MediaRead object for the cover photo.
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
        return usage.media.to_media_read()
