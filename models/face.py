from sqlmodel import Field, Column, JSON, Integer, Relationship, ForeignKey
from typing import Optional, List
from models.core import AppBaseModel


class FaceMatch(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("event.id", name="fk_facematch_event_id", ondelete="SET NULL"),
            nullable=False,
        )
    )
    media_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("media.id", name="fk_facematch_media_id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    embedding_index: int  # index in embeddings list
    matched_user_id: Optional[int] = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", name="fk_facematch_user_id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    distance: float
    is_participant: bool = Field(default=False)


class UnknownFaceCluster(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey(
                "event.id", name="fk_unknownfacecluster_event_id", ondelete="SET NULL"
            ),
        ),
    )
    cluster_label: str  # e.g., "cluster_1"
    embeddings: List[List[float]] = Field(sa_column=Column(JSON))
