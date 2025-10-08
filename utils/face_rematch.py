from models import (
    MediaEmbedding,
    MediaUsage,
    EventParticipant,
    FaceMatch,
    ContentOwnerType,
    MediaUsageType,
)
from sqlmodel import select
from scipy.spatial.distance import cosine
from socket_io import sio


async def retroactive_match_user_in_event(
    session, user_id: int, event_id: int, threshold: float = 0.6
):
    """Match a user against unmatched faces in a specific event."""

    # Get user's profile embedding
    user_embedding = session.exec(
        select(MediaEmbedding)
        .join(MediaUsage, MediaUsage.media_id == MediaEmbedding.media_id)
        .where(
            MediaUsage.owner_type == ContentOwnerType.USER,
            MediaUsage.owner_id == user_id,
            MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
            MediaEmbedding.status == "completed",
        )
    ).first()

    if not user_embedding or not user_embedding.embeddings:
        print(f"[RetroMatch] No embedding found for user {user_id}")
        return

    user_emb = user_embedding.embeddings[0]

    # Find unmatched faces in this event
    unmatched_faces = session.exec(
        select(FaceMatch, MediaEmbedding)
        .join(MediaEmbedding, MediaEmbedding.media_id == FaceMatch.media_id)
        .where(
            FaceMatch.event_id == event_id,
            FaceMatch.matched_user_id == None,
            MediaEmbedding.status == "completed",
        )
    ).all()

    matched_count = 0
    matched_media_ids = set()

    for face_match, media_embedding in unmatched_faces:
        if not media_embedding.embeddings:
            continue

        if face_match.embedding_index >= len(media_embedding.embeddings):
            continue

        face_emb = media_embedding.embeddings[face_match.embedding_index]
        distance = cosine(user_emb, face_emb)

        if distance < threshold:
            # Update the face match
            face_match.matched_user_id = user_id
            face_match.distance = float(distance)
            face_match.is_participant = True
            matched_count += 1
            matched_media_ids.add(face_match.media_id)

    session.commit()

    # Notify user about their matches
    if matched_count > 0:
        await sio.emit(
            "notification",
            {
                "type": "retroactive_matches_found",
                "event_id": event_id,
                "matched_count": matched_count,
                "media_count": len(matched_media_ids),
            },
            room=f"user:{user_id}",
        )

        print(
            f"[RetroMatch] Found {matched_count} matches for user {user_id} in event {event_id}"
        )
    else:
        print(f"[RetroMatch] No matches found for user {user_id} in event {event_id}")


async def retroactive_match_all_events(session, user_id: int, threshold: float = 0.6):
    """Match a user against unmatched faces in all events they participate in."""

    # Get user's profile embedding
    user_embedding = session.exec(
        select(MediaEmbedding)
        .join(MediaUsage, MediaUsage.media_id == MediaEmbedding.media_id)
        .where(
            MediaUsage.owner_type == ContentOwnerType.USER,
            MediaUsage.owner_id == user_id,
            MediaUsage.usage_type == MediaUsageType.PROFILE_PICTURE,
            MediaEmbedding.status == "completed",
        )
    ).first()

    if not user_embedding or not user_embedding.embeddings:
        print(f"[RetroMatch] No embedding found for user {user_id}")
        return

    user_emb = user_embedding.embeddings[0]

    # Get all events this user is approved for
    event_ids = [
        ep.event_id
        for ep in session.exec(
            select(EventParticipant.event_id).where(
                EventParticipant.user_id == user_id,
                EventParticipant.status == "approved",
            )
        ).all()
    ]

    if not event_ids:
        print(f"[RetroMatch] User {user_id} has no approved event participations")
        return

    # Find all unmatched faces in these events
    unmatched_faces = session.exec(
        select(FaceMatch, MediaEmbedding)
        .join(MediaEmbedding, MediaEmbedding.media_id == FaceMatch.media_id)
        .where(
            FaceMatch.event_id.in_(event_ids),
            FaceMatch.matched_user_id == None,
            MediaEmbedding.status == "completed",
        )
    ).all()

    total_matched = 0
    events_with_matches = {}

    for face_match, media_embedding in unmatched_faces:
        if not media_embedding.embeddings:
            continue

        if face_match.embedding_index >= len(media_embedding.embeddings):
            continue

        face_emb = media_embedding.embeddings[face_match.embedding_index]
        distance = cosine(user_emb, face_emb)

        if distance < threshold:
            # Update the face match
            face_match.matched_user_id = user_id
            face_match.distance = float(distance)
            face_match.is_participant = True
            total_matched += 1

            # Track matches per event
            if face_match.event_id not in events_with_matches:
                events_with_matches[face_match.event_id] = 0
            events_with_matches[face_match.event_id] += 1

    session.commit()

    # Notify user about their matches
    if total_matched > 0:
        await sio.emit(
            "notification",
            {
                "type": "retroactive_matches_found_all",
                "matched_count": total_matched,
                "events_count": len(events_with_matches),
                "events": [
                    {"event_id": eid, "match_count": count}
                    for eid, count in events_with_matches.items()
                ],
            },
            room=f"user:{user_id}",
        )

        print(
            f"[RetroMatch] Found {total_matched} matches for user {user_id} across {len(events_with_matches)} events"
        )
    else:
        print(f"[RetroMatch] No matches found for user {user_id}")
