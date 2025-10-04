from sqlmodel import Field, Column, JSON
from typing import Optional, List
from models.core import AppBaseModel


class FaceMatch(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="event.id")
    media_id: int = Field(foreign_key="media.id")
    embedding_index: int  # index in embeddings list

    matched_user_id: Optional[int] = Field(foreign_key="user.id")
    distance: float

    is_participant: bool = Field(default=False)


class UnknownFaceCluster(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(foreign_key="event.id")
    cluster_label: str  # e.g. "cluster_1"
    embeddings: List[List[float]] = Field(sa_column=Column(JSON))
