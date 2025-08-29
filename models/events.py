from datetime import datetime
from sqlmodel import DateTime, SQLModel, Field, Relationship, Column, JSON
from typing import Optional, List
from models.core import AppBaseModel


# Types
class EventBase(SQLModel):
    name: str = Field(index=True, unique=True)
    location: str
    description: str
    start_time: datetime
    end_time: datetime
    privacy: str
    cover_photo: List[str]


class EventCreate(EventBase):
    pass


# Models
class Event(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    location: str
    description: str
    start_time: datetime = Field(sa_column=Column(DateTime))
    end_time: datetime = Field(sa_column=Column(DateTime))
    privacy: str
    cover_photo: List[str] = Field(default=[], sa_column=Column(JSON))

    created_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_by: Optional["User"] = Relationship(back_populates="events")
