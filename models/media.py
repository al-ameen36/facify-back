from sqlmodel import Field, Relationship, Column, String, JSON, SQLModel
from typing import Optional, List
from models.core import AppBaseModel, ContentOwnerType, MediaType, MediaUsageType


# Response models
class MediaRead(SQLModel):
    """Media response model with base64 data for frontend"""
    id: int
    filename: str
    original_filename: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    duration: Optional[float] = None
    uploaded_by_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    # Base64 encoded media data for frontend
    data: Optional[str] = None


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


class MediaEmbedding(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    media_id: int = Field(foreign_key="media.id")
    model_name: str
    embedding: List[float] = Field(sa_column=Column(JSON))

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

    def get_base64_data(self, user: "User", drive_service=None) -> Optional[str]:
        """
        Returns the media as a base64 string ready for frontend.
        Returns None if fetching fails.
        """
        if not self.external_id:
            return None

        try:
            if not drive_service:
                from utils.drive import get_drive_service
                drive_service = get_drive_service(user)

            from googleapiclient.http import MediaIoBaseDownload
            import io
            import base64

            request = drive_service.files().get_media(fileId=self.external_id)
            file_stream = io.BytesIO()
            downloader = MediaIoBaseDownload(file_stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            file_stream.seek(0)
            encoded = base64.b64encode(file_stream.read()).decode()
            return f"data:{self.mime_type};base64,{encoded}"
        except Exception:
            # Fail gracefully
            return None

    def to_media_read(self, user: "User", drive_service=None) -> "MediaRead":
        """Convert Media to MediaRead with base64 data"""
        return MediaRead(
            id=self.id,
            filename=self.filename,
            original_filename=self.original_filename,
            file_size=self.file_size,
            mime_type=self.mime_type,
            duration=self.duration,
            uploaded_by_id=self.uploaded_by_id,
            created_at=self.created_at.isoformat() if self.created_at else None,
            updated_at=self.updated_at.isoformat() if self.updated_at else None,
            data=self.get_base64_data(user, drive_service)
        )
