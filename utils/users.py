from datetime import datetime, timedelta
from typing import Annotated, Optional
import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
from dotenv import load_dotenv
import os
from jose import JWTError
from fastapi_mail import MessageSchema
from models import User
from db import get_session

load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY", "insecure-default-key")
ALGORITHM = os.environ.get("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

active_refresh_tokens = set()


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_user_by_id(session: Session, user_id: int) -> Optional[User]:
    return session.exec(select(User).where(User.id == user_id)).first()


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    return session.exec(select(User).where(User.username == username)).first()


def get_user_by_email(session: Session, email: str) -> Optional[User]:
    return session.exec(select(User).where(User.email == email)).first()


def create_user(
    session: Session, username: str, email: str, full_name: str, password: str
) -> User:
    user = User(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
        disabled=False,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate_user(session: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(session, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(sub: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": sub, "type": "access", "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(sub: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (
        expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    payload = {"sub": sub, "type": "refresh", "exp": expire, "iat": datetime.utcnow()}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    active_refresh_tokens.add(token)
    return token


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except PyJWTError:
        return None


def verify_refresh_token(token: str) -> Optional[str]:
    if token not in active_refresh_tokens:
        return None
    payload = decode_token(token)
    if not payload or payload.get("type") != "refresh":
        return None
    return payload.get("sub")


def revoke_refresh_token(token: str):
    active_refresh_tokens.discard(token)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Session = Depends(get_session),
) -> User:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token: no subject")

    user = get_user_by_username(session, username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def send_password_reset_email(email: str, token: str):
    reset_url = f"{os.environ.get("RESET_PASSWORD_URL")}?token={token}"
    message = MessageSchema(
        subject="Password Reset Request",
        recipients=[email],
        body=f"Click the link to reset your password: {reset_url}",
        subtype="plain",
    )
    # fm = FastMail(conf)
    # await fm.send_message(message)


def verify_password_reset_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


def update_user_password(session: Session, user: User, new_password: str):
    user.hashed_password = pwd_context.hash(new_password)
    session.add(user)
    session.commit()


def get_user_by_provider_id(
    session: Session, provider: str, provider_id: str
) -> Optional[User]:
    return session.exec(
        select(User).where(User.provider == provider, User.provider_id == provider_id)
    ).first()


def authenticate_user(
    session: Session, email: str, password: str, checkVerified: bool = False
) -> Optional[User]:
    user = get_user_by_email(session, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    if checkVerified and not user.is_verified:
        raise HTTPException(status_code=403, detail="Email not verified")
    return user


async def send_verification_email(email: str, token: str):
    verify_url = f"{os.environ.get('VERIFY_EMAIL_URL')}?token={token}"
    message = MessageSchema(
        subject="Verify Your Email",
        recipients=[email],
        body=f"Click the link to verify your email: {verify_url}",
        subtype="plain",
    )
    # fm = FastMail(conf)
    # await fm.send_message(message)
