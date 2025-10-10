import sys
import os
from utils.face import generate_embeddings_background
from workers import app
from db import get_session
from utils.face_rematch import (
    retroactive_match_user_in_event,
    retroactive_match_all_events,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@app.task(name="tasks.face.embed_media", bind=True, max_retries=3)
async def embed_media(self, media_id: int, image_url: str):
    with next(get_session()) as session:
        try:
            await generate_embeddings_background(session, media_id, image_url)
        except Exception as e:
            print(f"Task failed: {e}")
            self.retry(exc=e, countdown=20)


@app.task(name="tasks.face.retroactive_match_task", bind=True, max_retries=3)
async def retroactive_match_task(self, user_id: int, event_id: int):
    """Match a user against unmatched faces in a specific event."""

    with next(get_session()) as session:
        try:
            await retroactive_match_user_in_event(session, user_id, event_id)
        except Exception as e:
            print(f"Task failed: {e}")
            self.retry(exc=e, countdown=20)


@app.task(name="tasks.face.retroactive_match_all_events_task", bind=True, max_retries=3)
async def retroactive_match_all_events_task(self, user_id: int):
    """Match a user against unmatched faces in all their events."""

    with next(get_session()) as session:
        try:
            await retroactive_match_all_events(session, user_id)
        except Exception as e:
            print(f"Task failed: {e}")
            self.retry(exc=e, countdown=20)
