from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship, Column, String, Boolean
from typing import Optional, List
from models.media import ContentOwnerType, MediaUsage, MediaUsageType, Media
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
    profile_picture: Optional["Media"] = None
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

    def get_profile_picture_url(self, session) -> Optional[str]:
        """Get current profile picture URL"""
        media = self.get_profile_picture_media(session)
        return media.url if media else None

    def to_user_read(self, session) -> "UserRead":
        """Convert User to UserRead with computed fields"""
        from sqlmodel import select, func
        
        # Import models to avoid circular imports
        from models.events import Event, EventParticipant
        from models.media import Media
        
        # Count hosted events
        num_hosted = session.exec(
            select(func.count(Event.id)).where(Event.created_by_id == self.id)
        ).one() or 0
        
        # Count joined events (approved participants)  
        num_joined = session.exec(
            select(func.count(EventParticipant.id)).where(
                EventParticipant.user_id == self.id,
                EventParticipant.status == "approved"
            )
        ).one() or 0
        
        # Count uploads
        num_uploads = session.exec(
            select(func.count(Media.id)).where(Media.uploaded_by_id == self.id)
        ).one() or 0
        
        # Count photos (images only)
        num_photos = session.exec(
            select(func.count(Media.id)).where(
                Media.uploaded_by_id == self.id,
                Media.mime_type.like("image/%")
            )
        ).one() or 0
        
        user_data = UserRead.model_validate(self)
        user_data.profile_picture = self.get_profile_picture_media(session)
        user_data.num_hosted = num_hosted
        user_data.num_joined = num_joined
        user_data.num_uploads = num_uploads
        user_data.num_photos = num_photos
        
        return user_data
