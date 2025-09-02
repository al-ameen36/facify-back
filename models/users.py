from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship, Column, String
from typing import Optional, List
from models.media import ContentOwnerType, MediaUsage, MediaUsageType


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

    events: List["Event"] = Relationship(back_populates="created_by")
    uploads: List["Media"] = Relationship(back_populates="uploaded_by")

    # Media helper methods
    def get_profile_picture(self, session) -> Optional["Media"]:
        """Get current profile picture"""
        usage = (
            session.query(MediaUsage)
            .filter(
                MediaUsage.owner_type == ContentOwnerType.USER,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
            )
            .first()
        )
        return usage.media if usage else None

    def get_my_uploads(self, session) -> List["Media"]:
        """Get all my uploads"""
        usages = (
            session.query(MediaUsage)
            .filter(
                MediaUsage.content_type == ContentOwnerType.USER,
                MediaUsage.object_id == self.id,
                MediaUsage.usage_type == MediaUsageType.GALLERY,
                MediaUsage.is_active == True,
            )
            .all()
        )
        return [usage.media for usage in usages if usage.media]

    def set_profile_picture(self, session, media: "Media"):
        """Set a new profile picture"""

        usage = (
            session.query(MediaUsage)
            .filter(
                MediaUsage.content_type == ContentOwnerType.USER,
                MediaUsage.object_id == self.id,
                MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
                MediaUsage.is_active == True,
            )
            .first()
        )

        usage.media = media

        session.commit()
        return usage
