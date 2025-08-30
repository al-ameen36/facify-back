from pydantic import EmailStr
from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from models.core import AppBaseModel


# Types
class UserBase(SQLModel):
    username: str = Field(index=True, unique=True)
    email: EmailStr = Field(index=True, unique=True)
    full_name: str
    disabled: bool = False

    num_joined: int = 0
    num_hosted: int = 0
    num_uploads: int = 0
    num_photos: int = 0


class UserCreate(UserBase):
    password: str


class UserRead(UserBase):
    id: int
    provider: Optional[str] = None
    provider_id: Optional[str] = None


class UserUpdate(SQLModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class Token(SQLModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: UserRead


class TokenData(SQLModel):
    username: Optional[str] = None


class RefreshTokenRequest(SQLModel):
    refresh_token: str


class ForgotPasswordRequest(SQLModel):
    email: EmailStr


class ResetPasswordRequest(SQLModel):
    token: str
    new_password: str


# Models
class User(AppBaseModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    email: str
    full_name: str
    hashed_password: str

    num_joined: Optional[int] = 0
    num_hosted: Optional[int] = 0
    num_uploads: Optional[int] = 0
    num_photos: Optional[int] = 0

    events: List["Event"] = Relationship(back_populates="created_by")
    uploads: list["Media"] = Relationship(back_populates="uploader")
