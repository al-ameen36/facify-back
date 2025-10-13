from .users import (
    User,
    UserCreate,
    UserRead,
    Token,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from .events import (
    Event,
    EventCreate,
    EventRead,
    EventParticipant,
    JoinEventRequest,
    EventCreateDB,
    ParticipantRead,
)
from .core import AppBaseModel, Pagination, SingleItemResponse, PaginatedResponse
from .media import (
    ContentOwnerType,
    Media,
    MediaUsage,
    MediaUsageType,
    MediaRead,
)
from .face import FaceCluster, FaceEmbedding
from .downloads import DownloadJob

# Rebuild models to resolve forward references
AppBaseModel.model_rebuild()
SingleItemResponse.model_rebuild()
PaginatedResponse.model_rebuild()
Pagination.model_rebuild()

User.model_rebuild()
Token.model_rebuild()
UserRead.model_rebuild()
UserCreate.model_rebuild()
RefreshTokenRequest.model_rebuild()
ForgotPasswordRequest.model_rebuild()
ResetPasswordRequest.model_rebuild()

Event.model_rebuild()
EventRead.model_rebuild()
EventCreate.model_rebuild()
EventCreateDB.model_rebuild()
JoinEventRequest.model_rebuild()
EventParticipant.model_rebuild()
ParticipantRead.model_rebuild()

Media.model_rebuild()
MediaUsage.model_rebuild()
MediaRead.model_rebuild()

FaceEmbedding.model_rebuild()
FaceCluster.model_rebuild()

DownloadJob.model_rebuild()


# Export models
__all__ = [
    # Core
    "AppBaseModel",
    "SingleItemResponse",
    "PaginatedResponse",
    "Pagination",
    # Users
    "User",
    "Token",
    "UserRead",
    "UserCreate",
    "RefreshTokenRequest",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    # Events
    "Event",
    "EventCreate",
    "EventRead" "EventCreateDB",
    "JoinEventRequest",
    "EventParticipant",
    "ParticipantRead",
    # Media
    "Media",
    "MediaRead",
    "MediaUsage",
    "MediaUsageType",
    "ContentOwnerType",
    # Face
    "FaceEmbedding",
    "FaceCluster",
    # Downloads
    "DownloadJob",
]
