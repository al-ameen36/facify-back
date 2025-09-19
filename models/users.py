from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship, Column, String, Boolean
from typing import Optional, List
from models.media import ContentOwnerType, MediaUsage, MediaUsageType
from datetime import datetime


# Types (Pydantic models for API)
class UserBase(SQLModel):
    username: str
    email: EmailStr
    full_name: str


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: int
    profile_picture: Optional[str] = None
    is_drive_connected: Optional[bool] = False

    num_joined: int = 0
    num_hosted: int = 0
    num_uploads: int = 0
    num_photos: int = 0


class UserUpdate(SQLModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class Token(SQLModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserRead


class TokenData(SQLModel):
    username: Optional[str] = None


class RefreshTokenRequest(SQLModel):
    refresh_token: str


class ForgotPasswordRequest(SQLModel):
    email: EmailStr


class ResetPasswordRequest(SQLModel):
    token: str
    new_password: str


# Database Models
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String, unique=True, index=True))
    email: str = Field(sa_column=Column(String, unique=True, index=True))
    full_name: str
    hashed_password: str

    # One-to-one: user → face embedding
    face_embedding: Optional["MediaEmbedding"] = Relationship(back_populates="user")

    events: List["Event"] = Relationship(back_populates="created_by")
    uploads: List["Media"] = Relationship(back_populates="uploaded_by")
    joined_events: List["Event"] = Relationship(back_populates="participants")

    # Google OAuth tokens
    is_drive_connected: bool = Field(
        default=False, sa_column=Column(Boolean, default=False, nullable=False)
    )
    drive_access_token: Optional[str] = None
    drive_refresh_token: Optional[str] = None
    drive_token_expiry: Optional[datetime] = None

    # Media helper methods
    def get_profile_picture_media(self, session) -> Optional["Media"]:
        """Get current profile picture Media object"""
        from sqlmodel import select
        usage = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.USER,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
            )
        ).first()
        return usage.media if usage else None

    def get_profile_picture_base64(self, session, drive_service=None) -> Optional[str]:
        """Get profile picture as base64 string for frontend"""
        media = self.get_profile_picture_media(session)
        if not media or not media.external_id:
            return None

        try:
            if not drive_service:
                from utils.drive import get_drive_service
                drive_service = get_drive_service(self)

            from googleapiclient.http import MediaIoBaseDownload
            import io
            import base64

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

    def get_my_uploads(self, session) -> List["Media"]:
        """Get all my uploads"""
        from sqlmodel import select
        usages = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.USER,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.GALLERY,
            )
        ).all()
        return [usage.media for usage in usages if usage.media]

    def set_profile_picture(self, session, media: "Media"):
        """Set a new profile picture"""
        from sqlmodel import select
        
        # Find existing profile picture usage
        usage = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.USER,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
            )
        ).first()

        if usage:
            # Update existing usage
            usage.media_id = media.id
        else:
            # Create new usage
            usage = MediaUsage(
                owner_type=ContentOwnerType.USER,
                owner_id=self.id,
                usage_type=MediaUsageType.PROFILE_PICTURE,
                media_type="image",
                media_id=media.id,
            )
            session.add(usage)

        session.commit()
        return usage
