from celery import Celery
from celery_aio_pool.pool import AsyncIOPool
import socketio
from dotenv import load_dotenv
import os

load_dotenv()

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
SIO_REDIS_URL = os.getenv("SIO_REDIS_URL")


app = Celery(
    "embeddings",
    broker=f"{CELERY_BROKER_URL}/0",
    include=["tasks.face", "tasks.notifications", "tasks.downloads"],
)


app.conf.update(worker_pool=AsyncIOPool)

sio_manager = socketio.AsyncRedisManager(f"{SIO_REDIS_URL}/1")


async def emit_to_all(event: str, data: dict):
    """Helper to emit events to all connected clients"""
    await sio_manager.emit(event, data)


async def emit_to_user(user_id: str, event: str, data: dict):
    """Helper to emit events to a specific room"""

    await sio_manager.emit(
        "notification",
        {
            "type": event,
            "data": data,
        },
        room=f"user:{user_id}",
    )
