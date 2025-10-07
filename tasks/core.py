import sys
import os
from celery import Celery
from celery_aio_pool.pool import AsyncIOPool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


app = Celery(
    "embeddings",
    broker="redis://localhost:6379/0",
    include=["tasks.face", "tasks.notifications"],
)


app.conf.update(worker_pool=AsyncIOPool)
