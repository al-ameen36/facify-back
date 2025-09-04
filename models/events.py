from datetime import datetime
from models.media import MediaUsage
from sqlmodel import SQLModel, Field, Relationship, select
from typing import List, Optional
from models.core import AppBaseModel, ContentOwnerType, MediaUsageType
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


# Types
class EventBase(SQLModel):
    name: str = Field(index=True, unique=True)
    location: str
    description: str
    start_time: datetime
    end_time: datetime
    privacy: str


class EventRead(EventBase):
    id: int
    updated_at: datetime
    created_by_id: int
    created_at: datetime
    cover_photo: Optional[str] = None


class EventCreate(EventBase):
    pass


class JoinEventRequest(SQLModel):
    secret: str


# Models
class EventParticipant(SQLModel, table=True):
    """Junction table for many-to-many relationship between events and users"""

    event_id: int = Field(foreign_key="event.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)


class Event(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    location: str
    start_time: datetime
    end_time: datetime
    privacy: str
    secret: Optional[str] = Field(default_factory=generate_event_secret)
    cover_photo: Optional[str] = None

    # Relationships
    created_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_by: Optional["User"] = Relationship(back_populates="events")
    participants: List["User"] = Relationship(back_populates="joined_events")

    def add_participant(self, session, user: "User"):
        """Add a participant to the event"""
        participant = EventParticipant(event_id=self.id, user_id=user.id)
        session.add(participant)
        session.commit()
        return participant

    def remove_participant(self, session, user: "User"):
        """Remove a participant from the event"""
        participant = (
            session.query(EventParticipant)
            .filter(
                EventParticipant.event_id == self.id,
                EventParticipant.user_id == user.id,
            )
            .first()
        )
        if participant:
            session.delete(participant)
            session.commit()
        return participant

    def get_cover_photo(self, session) -> Optional["Media"]:
        usage = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.EVENT,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.COVER_PHOTO,
            )
        ).first()
        return usage.media if usage else None
