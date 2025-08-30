from sqlmodel import Field, Column, Relationship, String
from typing import Optional
from .core import AppBaseModel


class Media(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str
    type: str = Field(sa_column=Column(String, index=True))
    owner_id: int = Field(index=True)
    owner_type: str = Field(sa_column=Column(String, index=True))
    uploaded_by: Optional[int] = Field(default=None, foreign_key="user.id")

    uploader: Optional["User"] = Relationship(back_populates="uploads")
