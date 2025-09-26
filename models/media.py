from sqlmodel import Field, Relationship, Column, String, JSON, SQLModel, Index
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
    uploaded_by_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MediaUsage(AppBaseModel, table=True):
    """Generic relationship table linking Media to any content type"""
    
    __table_args__ = (
        # Composite indexes for common query patterns
        Index("idx_media_usage_owner_usage", "owner_type", "owner_id", "usage_type"),
        Index("idx_media_usage_media_usage", "media_id", "usage_type"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    owner_type: ContentOwnerType = Field(sa_column=Column(String, index=True))
    owner_id: int = Field(index=True)
    usage_type: MediaUsageType = Field(sa_column=Column(String, index=True))
    media_type: MediaType = Field(sa_column=Column(String, index=True))

    # Relationship to Media
    media_id: int = Field(foreign_key="media.id", unique=True)
    media: "Media" = Relationship(back_populates="usage")


class MediaEmbedding(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    media_id: int = Field(foreign_key="media.id")
    model_name: str
    embeddings: Optional[List[List[float]]] = Field(default=None, sa_column=Column(JSON))
    
    # Background processing status
    status: Optional[str] = Field(default="pending")  # pending, processing, completed, failed
    processed_at: Optional[str] = None
    error_message: Optional[str] = None

    # For user enrollment (one-to-one)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    user: Optional["User"] = Relationship(back_populates="face_embedding")

    # Normal case → event photo with multiple embeddings
    media: "Media" = Relationship(back_populates="embeddings")


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

    # Relationships
    uploaded_by_id: Optional[int] = Field(default=None, foreign_key="user.id")
    uploaded_by: Optional["User"] = Relationship(back_populates="uploads")
    usage: "MediaUsage" = Relationship(back_populates="media")

    # NEW: embeddings for all faces in this media
    embeddings: List["MediaEmbedding"] = Relationship(back_populates="media")

    @property
    def url(self) -> str:
        return f"/uploads/{self.id}/file"



    def to_media_read(self) -> "MediaRead":
        """Convert Media to MediaRead"""
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
        )
