import sys
import os
from utils.face import generate_embeddings_background
from tasks.core import app

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@app.task(name="tasks.face.embed_media", bind=True, max_retries=3)
async def embed_media(self, media_id: int, image_url: str):
    try:
        await generate_embeddings_background(media_id, image_url)
    except Exception as e:
        print(f"Task failed: {e}")
        self.retry(exc=e, countdown=60)
