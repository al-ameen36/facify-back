from celery import Celery
from celery_aio_pool.pool import AsyncIOPool
import socketio


app = Celery(
    "embeddings",
    broker="redis://localhost:6379/0",
    include=["tasks.face", "tasks.notifications", "tasks.downloads"],
)


app.conf.update(worker_pool=AsyncIOPool)

sio_manager = socketio.AsyncRedisManager("redis://localhost:6379/1")


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
