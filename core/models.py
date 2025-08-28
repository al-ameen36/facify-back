from typing import Generic, TypeVar, List, Optional
from sqlmodel import SQLModel

DataT = TypeVar("DataT")


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
