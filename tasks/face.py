from utils.face import generate_embeddings_background
from workers import app
from db import get_session


@app.task(name="tasks.face.embed_media", bind=True, max_retries=3)
def embed_media(self, media_id: int, image_url: str):
    with next(get_session()) as session:
        try:
            generate_embeddings_background(session, media_id, image_url)
        except Exception as e:
            print(f"Task failed: {e}")
            self.retry(exc=e, countdown=20)
