from typing import Generic, TypeVar, List, Optional
from sqlmodel import SQLModel, Field
from datetime import datetime, timezone

DataT = TypeVar("DataT")


def utc_now() -> datetime:
    """Return current UTC datetime"""
    return datetime.now(timezone.utc)


class Pagination(SQLModel):
    total: int
    page: int
    per_page: int
    total_pages: int


class PaginatedResponse(SQLModel, Generic[DataT]):
    message: str
    data: List[DataT]
    pagination: Pagination


class SingleItemResponse(SQLModel, Generic[DataT]):
    message: str
    data: Optional[DataT]


class AppBaseModel(SQLModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = Field(default_factory=utc_now)
    updated_at: Optional[datetime] = Field(default_factory=utc_now)
