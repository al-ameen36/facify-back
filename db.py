from sqlmodel import create_engine, Session
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")

# Create SQLModel engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)


def create_db_and_tables():
    """No-op when using Alembic migrations"""
    print("Skipping auto table creation; using Alembic for migrations.")


def get_session():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
