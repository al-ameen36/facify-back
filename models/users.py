from pydantic import EmailStr
from sqlmodel import (
    SQLModel,
    Field,
    Relationship,
    Column,
    String,
    func,
    select,
)
from typing import Optional, List
from models.media import Media, MediaRead
from models.events import EventParticipant


# --- Pydantic Models ---
class UserBase(SQLModel):
    username: str
    email: EmailStr
    full_name: str


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: int
    profile_picture: Optional[MediaRead] = None
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


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String, unique=True, index=True))
    email: str = Field(sa_column=Column(String, unique=True, index=True))
    full_name: str
    hashed_password: str

    # One-to-one: user → face embedding (latest profile picture)
    face_clusters: List["FaceCluster"] = Relationship(back_populates="user")

    # One-to-many: user → events (created by)
    events: List["Event"] = Relationship(back_populates="created_by")

    # One-to-many: user → media (uploads)
    uploads: List[Media] = Relationship(back_populates="uploaded_by")

    # Many-to-many: user ↔ events (participants)
    joined_events: List["Event"] = Relationship(
        back_populates="participants", link_model=EventParticipant
    )

    # --- Helper Methods ---
    def get_profile_picture_media(self, session) -> Optional[MediaRead]:
        """Get current profile picture MediaRead object"""
        from models import MediaUsage, ContentOwnerType, MediaUsageType

        usage = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.USER,
                MediaUsage.owner_id == self.id,
                MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
                MediaUsage.tags == "center",
            )
        ).first()
        return usage.media.to_media_read() if usage and usage.media else None

    def to_user_read(self, session) -> UserRead:
        """Convert User to UserRead with computed fields"""
        from models import (
            Event,
            EventParticipant,
            Media,
            MediaUsage,
            MediaUsageType,
            FaceCluster,
            FaceEmbedding,
        )

        # Count hosted events
        num_hosted = (
            session.exec(
                select(func.count(Event.id)).where(Event.created_by_id == self.id)
            ).one()
            or 0
        )

        # Count joined events (approved participants)
        num_joined = (
            session.exec(
                select(func.count(EventParticipant.id)).where(
                    EventParticipant.user_id == self.id,
                    EventParticipant.status == "approved",
                )
            ).one()
            or 0
        )

        num_uploads = (
            session.exec(
                select(func.count(MediaUsage.id))
                .join(Media)
                .where(
                    Media.uploaded_by_id == self.id,
                    MediaUsage.usage_type == MediaUsageType.GALLERY,
                )
            ).one()
            or 0
        )

        # Count photos where user's face was detected (through face clusters)
        num_photos = (
            session.exec(
                select(func.count(FaceEmbedding.id.distinct()))
                .join(FaceCluster)
                .where(
                    FaceCluster.user_id == self.id,
                    FaceEmbedding.status
                    == "completed",  # Only completed face detections
                )
            ).one()
            or 0
        )

        user_data = UserRead.model_validate(self)
        user_data.profile_picture = self.get_profile_picture_media(session)
        user_data.num_hosted = num_hosted
        user_data.num_joined = num_joined
        user_data.num_uploads = num_uploads
        user_data.num_photos = num_photos
        return user_data
