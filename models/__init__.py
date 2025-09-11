from .users import (
    User,
    FaceEmbedding,
    UserCreate,
    UserRead,
    Token,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from .events import Event, EventCreate, EventParticipant, JoinEventRequest
from .core import AppBaseModel, Pagination, SingleItemResponse, PaginatedResponse
from .media import ContentOwnerType, Media, MediaUsage, MediaUsageType


# Rebuild models to resolve forward references
AppBaseModel.model_rebuild()
SingleItemResponse.model_rebuild()
PaginatedResponse.model_rebuild()
Pagination.model_rebuild()

User.model_rebuild()
FaceEmbedding.model_rebuild()
Token.model_rebuild()
UserRead.model_rebuild()
UserCreate.model_rebuild()
RefreshTokenRequest.model_rebuild()
ForgotPasswordRequest.model_rebuild()
ResetPasswordRequest.model_rebuild()

Event.model_rebuild()
EventCreate.model_rebuild()
JoinEventRequest.model_rebuild()
EventParticipant.model_rebuild()

Media.model_rebuild()
MediaUsage.model_rebuild()


# Export models
__all__ = [
    # Core
    "AppBaseModel",
    "SingleItemResponse",
    "PaginatedResponse",
    "Pagination",
    # Users
    "User",
    "FaceEmbedding",
    "Token",
    "UserRead",
    "UserCreate",
    "RefreshTokenRequest",
    "ForgotPasswordRequest",
    "ResetPasswordRequest",
    # Events
    "Event",
    "EventCreate",
    "JoinEventRequest",
    "EventParticipant",
    # Media
    "Media",
    "MediaUsage",
    "MediaUsageType",
    "ContentOwnerType",
]
