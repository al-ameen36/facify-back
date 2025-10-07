from celery import Celery

app = Celery("embeddings", broker="redis://localhost:6379/0", include=["tasks.face"])

from tasks.face import embed_media
