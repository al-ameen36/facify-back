import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery import Celery
from utils.face import generate_embeddings_background

app = Celery("embeddings", broker="redis://localhost:6379/0")


@app.task(name="tasks.face.embed_media", bind=True, max_retries=3)
def embed_media(self, media_id: int, image_url: str):
    try:
        generate_embeddings_background(media_id, image_url)
    except Exception as e:
        print(f"Task failed: {e}")
        self.retry(exc=e, countdown=60)
