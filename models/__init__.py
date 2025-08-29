from .users import (
    User,
    UserCreate,
    UserRead,
    Token,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from .events import Event, EventCreate
from .core import AppBaseModel, SingleItemResponse, PaginatedResponse

# Rebuild models to resolve forward references
AppBaseModel.model_rebuild()
SingleItemResponse.model_rebuild()
PaginatedResponse.model_rebuild()

User.model_rebuild()
Token.model_rebuild()
UserRead.model_rebuild()
UserCreate.model_rebuild()
RefreshTokenRequest.model_rebuild()
ForgotPasswordRequest.model_rebuild()
ResetPasswordRequest.model_rebuild()

Event.model_rebuild()
EventCreate.model_rebuild()


# Export models
__all__ = [
    # Core
    "AppBaseModel",
    "SingleItemResponse",
    "PaginatedResponse",
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
]
