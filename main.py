from fastapi import FastAPI
from contextlib import asynccontextmanager
from db import create_db_and_tables, get_session
from models.users import User
from utils.users import get_password_hash
from sqlmodel import select
from dotenv import load_dotenv
from routes.users import router as auth_router
import os
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@email.com")
DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    create_db_and_tables()
    print("Database tables created")

    # Use `next(get_session())` to get the actual session
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
            email="user@email.com",
            username="user",
            full_name="user",
            hashed_password=hashed_password,
            is_admin=False,
            is_active=True,
        )
        session.add(admin_user)
        session.add(test_user)
        session.commit()
        print(f"✅ Users created: {ADMIN_EMAIL}")
    else:
        print("✅ Users already exists")

    yield
    print("Application shutting down")


app = FastAPI(
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(auth_router)


@app.get("/")
async def root():
    return {"message": "Authentication API with SQLModel is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
