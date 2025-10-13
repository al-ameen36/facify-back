from datetime import datetime, timezone
from typing import List, Optional
from sqlmodel import (
    SQLModel,
    Field,
    Column,
    JSON,
    Integer,
    ForeignKey,
    Relationship,
    String,
)


class FaceCluster(SQLModel, table=True):
    """
    A group of similar faces (one person identity).
    If matched to a known user, we can link user_id.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    label: Optional[str] = Field(default=None, index=True)
    centroid: List[float] = Field(sa_column=Column(JSON))
    user_id: Optional[int] = Field(
        sa_column=Column(
            Integer,
            ForeignKey("user.id", name="fk_facecluster_user_id", ondelete="SET NULL"),
            nullable=True,
        )
    )

    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    faces: List["FaceEmbedding"] = Relationship(back_populates="cluster")
    user: Optional["User"] = Relationship(back_populates="face_clusters")
    event_id: Optional[int] = Field(
        sa_column=Column(
            Integer,
            ForeignKey("event.id", name="fk_facecluster_event_id", ondelete="CASCADE"),
            nullable=True,
        )
    )
    event: Optional["Event"] = Relationship(back_populates="face_clusters")


class FaceEmbedding(SQLModel, table=True):
    """
    Stores individual face embeddings from photos/videos.
    Each embedding belongs to a specific media and optional cluster.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    media_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey(
                "media.id", name="fk_faceembedding_media_id", ondelete="CASCADE"
            ),
            nullable=False,
        )
    )
    embedding: List[float] = Field(sa_column=Column(JSON))
    # Background processing status
    status: Optional[str] = Field(
        sa_column=Column(
            String,
            server_default="pending",
            nullable=False,
            default="pending",
        ),
    )  # pending, processing, completed, failed
    facial_area: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    cluster_id: Optional[int] = Field(
        sa_column=Column(
            Integer,
            ForeignKey(
                "facecluster.id",
                name="fk_faceembedding_cluster_id",
                ondelete="SET NULL",
            ),
            nullable=True,
        )
    )
    tags: Optional[str] = Field(default=None, sa_column=Column(String))
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    cluster: Optional[FaceCluster] = Relationship(back_populates="faces")
    media: "Media" = Relationship(back_populates="face_embeddings")
