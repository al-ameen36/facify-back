import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlmodel import select
from db import create_db_and_tables, get_session
from models import User
from utils.users import get_password_hash
from routes.users import router as auth_router
from routes.events import router as event_router
from routes.media import router as media_router
from routes.face import router as face_router
from routes.downloads import router as downloads_router
import socketio
from socket_io import sio


load_dotenv()

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@email.com")
DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    print("✅ Database tables created")

    session = next(get_session())
    existing_admin = session.exec(select(User).where(User.email == ADMIN_EMAIL)).first()

    if not existing_admin:
        hashed_password = get_password_hash(DEFAULT_PASSWORD)
        admin_user = User(
            email=ADMIN_EMAIL,
            username="admin",
            full_name="admin",
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True,
        )
        test_user = User(
            email="muhdabdullahi361@gmail.com",
            username="maa",
            full_name="Muhammad Al-Ameen Abdullahi",
            hashed_password=hashed_password,
            is_active=True,
        )
        session.add(admin_user)
        session.add(test_user)
        session.commit()
        print("✅ Default users created")
    else:
        print("✅ Users already exist")

    try:
        yield
    finally:
        print("Application shutting down")


# 1️⃣ Create FastAPI app
fastapi_app = FastAPI(
    version="1.0.0",
    lifespan=lifespan,
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2️⃣ Include routes
fastapi_app.include_router(auth_router)
fastapi_app.include_router(event_router)
fastapi_app.include_router(media_router)
fastapi_app.include_router(face_router)
fastapi_app.include_router(downloads_router)


# 3️⃣ REST routes
@fastapi_app.get("/")
async def root():
    return {"message": "Authentication API with SQLModel is running!"}


@fastapi_app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 5️⃣ Combine FastAPI + Socket.IO
app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
