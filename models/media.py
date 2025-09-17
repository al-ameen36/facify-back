from sqlmodel import Field, Relationship, Column, String
from typing import Optional
from models.core import AppBaseModel, ContentOwnerType, MediaType, MediaUsageType


class MediaUsage(AppBaseModel, table=True):
    """Generic relationship table linking Media to any content type"""

    id: Optional[int] = Field(default=None, primary_key=True)

    owner_type: ContentOwnerType = Field(sa_column=Column(String, index=True))
    owner_id: int = Field(index=True)
    usage_type: MediaUsageType = Field(sa_column=Column(String, index=True))
    media_type: MediaType = Field(sa_column=Column(String, index=True))

    # Relationship to Media
    media_id: int = Field(foreign_key="media.id", unique=True)
    media: "Media" = Relationship(back_populates="usage")


class Media(AppBaseModel, table=True):
    """Generic Media model that can be used by any content type"""

    id: Optional[int] = Field(default=None, primary_key=True)
    url: str
    filename: str = Field(default="unknown")
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    duration: Optional[float] = None
    external_id: str

    # Relationships
    uploaded_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    uploaded_by: Optional["User"] = Relationship(back_populates="uploads")
    usage: "MediaUsage" = Relationship(back_populates="media")
