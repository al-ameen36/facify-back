from sqlmodel import select
from models import MediaEmbedding, Media
from utils.face import generate_embeddings_background
from db import get_session
import asyncio


async def retry_failed_embeddings():
    """Retry failed embeddings periodically."""
    print("🔁 Checking for failed embeddings...")

    session = next(get_session())

    # ✅ Only select FAILED embeddings, not ones already completed
    failed_embeddings = session.exec(
        select(MediaEmbedding)
        .where(MediaEmbedding.status == "failed")
        .where(MediaEmbedding.retry_count < 3)
    ).all()

    for embedding in failed_embeddings:
        # ✅ Double-check file accessibility (optional)
        media = session.get(Media, embedding.media_id)
        if not media or not media.external_url:
            print(f"⚠️ Skipping media {embedding.media_id}, URL missing")
            continue

        # ✅ Ensure status hasn’t changed since last fetch
        session.refresh(embedding)
        if embedding.status != "failed":
            print(f"⏭ Skipping media {embedding.media_id}, status = {embedding.status}")
            continue

        try:
            print(f"🚀 Retrying embedding for media {embedding.media_id}")
            embedding.status = "retrying"
            embedding.retry_count += 1
            session.commit()

            # ✅ Run async embedding task safely
            asyncio.create_task(
                generate_embeddings_background(embedding.media_id, media.external_url)
            )
        except Exception as e:
            print(f"❌ Failed to reprocess media {embedding.media_id}: {e}")
            embedding.error_message = str(e)
            session.commit()

    session.close()
