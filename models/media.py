from datetime import datetime
from sqlmodel import (
    Field,
    ForeignKey,
    Integer,
    Relationship,
    Column,
    String,
    SQLModel,
    Index,
)
from typing import Optional, List
from models.core import AppBaseModel, ContentOwnerType, MediaType, MediaUsageType


# Response models
class MediaRead(SQLModel):
    """Media response model for frontend"""

    id: int
    filename: str
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    url: str
    mime_type: Optional[str] = None
    duration: Optional[float] = None
    uploaded_by_id: int
    created_at: Optional[str | datetime] = None
    updated_at: Optional[str | datetime] = None
    face_count: Optional[int] = 0
    status: Optional[str] = None


class MediaUsage(AppBaseModel, table=True):
    """Generic relationship table linking Media to any content type"""

    __table_args__ = (
        # Composite indexes for common query patterns
        Index("idx_media_usage_owner_usage", "owner_type", "owner_id", "usage_type"),
        Index("idx_media_usage_media_usage", "media_id", "usage_type"),
        Index("idx_media_usage_approval", "owner_type", "approval_status"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_type: ContentOwnerType = Field(sa_column=Column(String, index=True))
    owner_id: int = Field(index=True)
    usage_type: MediaUsageType = Field(sa_column=Column(String, index=True))
    media_type: MediaType = Field(sa_column=Column(String, index=True))
    approval_status: Optional[str] = Field(
        default="pending", sa_column=Column(String, index=True)
    )
    approved_at: Optional[datetime] = None
    tags: Optional[str] = Field(default=None, sa_column=Column(String))

    # Relationship to Media
    media_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("media.id", name="fk_mediausage_media_id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    media: "Media" = Relationship(back_populates="usage")


class Media(AppBaseModel, table=True):
    """Generic Media model that can be used by any content type"""

    id: Optional[int] = Field(default=None, primary_key=True)
    external_url: str
    filename: str = Field(default="unknown")
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    duration: Optional[float] = None
    external_id: str
    face_count: Optional[int] = Field(default=0)
    tags: Optional[str] = Field(default=None, sa_column=Column(String))

    # Relationships
    uploaded_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    uploaded_by: Optional["User"] = Relationship(back_populates="uploads")
    usage: List["MediaUsage"] = Relationship(
        back_populates="media",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "passive_deletes": True,
        },
    )
    face_embeddings: List["FaceEmbedding"] = Relationship(
        back_populates="media",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "passive_deletes": True,
        },
    )

    def to_media_read(self) -> "MediaRead":
        """Convert Media to MediaRead"""
        # Get status from first usage if available
        status = self.usage[0].approval_status if self.usage else None

        return MediaRead(
            id=self.id,
            filename=self.filename,
            original_filename=self.original_filename,
            file_size=self.file_size,
            url=self.external_url,
            mime_type=self.mime_type,
            duration=self.duration,
            uploaded_by_id=self.uploaded_by_id,
            created_at=self.created_at.isoformat() if self.created_at else None,
            updated_at=self.updated_at.isoformat() if self.updated_at else None,
            face_count=self.face_count,
            status=status,
        )
