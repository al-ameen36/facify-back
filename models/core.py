from typing import Generic, TypeVar, List, Optional
from sqlmodel import SQLModel, Field
from enum import Enum
from datetime import datetime, timezone

DataT = TypeVar("DataT")


# Pagination
class Pagination(SQLModel):
    total: int
    page: int
    per_page: int
    total_pages: int


class PaginatedResponse(SQLModel, Generic[DataT]):
    message: str
    data: List[DataT]
    pagination: Pagination


class PaginatedSingleItemResponse(SQLModel, Generic[DataT]):
    message: str
    data: DataT
    pagination: Pagination


class SingleItemResponse(SQLModel, Generic[DataT]):
    message: str
    data: Optional[DataT]


class AppBaseModel(SQLModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# Media
class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class ContentOwnerType(str, Enum):
    USER = "user"
    EVENT = "event"


class MediaUsageType(str, Enum):
    PROFILE_PICTURE = "profile_picture"
    PROFILE_PICTURE_ANGLE = "profile_picture_angle"
    PROFILE_PICTURE_ARCHIVED = "profile_picture_archived"
    COVER_PHOTO = "cover_photo"
    GALLERY = "gallery"
