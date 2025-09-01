from datetime import datetime
from sqlmodel import DateTime, SQLModel, Field, Relationship, Column, JSON, String
from typing import Optional, List
from models.core import AppBaseModel
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
    cover_photo: List[str]
    secret: Optional[str] = None


class EventCreate(EventBase):
    pass


class JoinEventRequest(SQLModel):
    secret: str


# Models
class EventParticipant(AppBaseModel, table=True):
    event_id: int = Field(foreign_key="event.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)


class Event(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    location: str
    description: str
    start_time: datetime = Field(sa_column=Column(DateTime))
    end_time: datetime = Field(sa_column=Column(DateTime))
    privacy: str
    cover_photo: List[str] = Field(default=[], sa_column=Column(JSON))
    secret: str = Field(
        sa_column=Column(String, nullable=False, unique=True),
        default_factory=generate_event_secret,
    )

    created_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_by: Optional["User"] = Relationship(back_populates="events")

    participants: List["User"] = Relationship(
        back_populates="events_joined", link_model=EventParticipant
    )
