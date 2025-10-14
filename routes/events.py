from datetime import datetime, timezone
from typing import Optional
from fastapi import Depends, APIRouter, HTTPException, Query, Body
from sqlmodel import Session, func, select, SQLModel
from models import (
    EventRead,
    ParticipantRead,
    Event,
    EventCreate,
    EventCreateDB,
    User,
    PaginatedResponse,
    SingleItemResponse,
    JoinEventRequest,
    Pagination,
    EventParticipant,
)
from utils.events import (
    create_event,
    update_event,
    get_event_by_name,
    get_event_by_name,
)
from utils.users import get_current_user
from db import get_session
from tasks.notifications import send_notification


router = APIRouter(prefix="/events", tags=["event"])


@router.post("/{event_id}/join")
async def join_event(
    event_id: int,
    body: JoinEventRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Event Status/Time Guards: Prevent joining events that have already ended
    now = datetime.now(timezone.utc)
    end_time = event.end_time.replace(tzinfo=timezone.utc)
    if end_time < now:
        raise HTTPException(
            status_code=400, detail="Cannot join an event that has already ended"
        )

    # Event Privacy Guards: Check if event is accessible
    if not event.secret or event.secret != body.secret:
        raise HTTPException(status_code=403, detail="Invalid secret")
    # check if already a participant
    statement = select(EventParticipant).where(
        EventParticipant.event_id == event_id,
        EventParticipant.user_id == current_user.id,
    )
    existing = session.exec(statement).first()
    if existing:
        raise HTTPException(status_code=400, detail="Already a participant")

    # By default, participant status is "pending"
    # status = "approved" if event.auto_approve_participants else "pending"
    # status = "approved" if event.auto_approve_uploads
    participant = EventParticipant(
        event_id=event_id, user_id=current_user.id, status="pending"
    )

    session.add(participant)
    session.commit()

    # Notify Host
    participant_user = session.get(User, current_user.id)
    send_notification.delay(
        user_id=event.created_by_id,
        event="new_event_participant",
        data={
            "event_id": event.id,
            "participant_id": participant_user.id,
            "participant_name": participant_user.full_name,
        },
    )

    return SingleItemResponse(
        data=event,
        message=f"{current_user.username} joined the event (pending approval)",
    )


@router.post("", response_model=SingleItemResponse[EventRead])
async def add_event(
    event_data: EventCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    existing_event = session.exec(
        select(Event).where(Event.name == event_data.name)
    ).first()
    if existing_event:
        raise HTTPException(
            status_code=400, detail="An event with the name '{value}' already exists."
        )

    try:
        event = EventCreateDB(
            created_by_id=current_user.id,
            name=event_data.name,
            location=event_data.location,
            description=event_data.description,
            start_time=event_data.start_time,
            end_time=event_data.end_time,
            privacy=event_data.privacy,
        )
        event = create_event(session=session, event=event)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SingleItemResponse[Event](message="Event created successfully", data=event)


@router.patch("", response_model=SingleItemResponse[EventRead])
async def update_event_route(
    event_data: EventCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    event = get_event_by_name(session, event_data.name)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if event.created_by_id != current_user.id:
        raise HTTPException(status_code=401, detail="Not authorized")

    try:
        updated_event = update_event(
            session=session,
            event=event,
            update_data=event_data,
            created_by_id=event.created_by_id,
        )

        event_dict = updated_event.model_dump()
        event_dict["cover_photo"] = updated_event.get_cover_photo_media(session)
        event = EventRead(**event_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # --- Notify Participants ---
    from socket_io import sio

    # Fetch participant user IDs (except creator if you want)
    participant_ids = session.exec(
        select(EventParticipant.user_id).where(EventParticipant.event_id == event.id)
    ).all()

    for user_id in participant_ids:
        send_notification.delay(
            user_id=user_id,
            event="event_updated",
            data={
                "event_id": event.id,
                "event_name": event.name,
                "message": f"The event '{event.name}' has been updated.",
            },
        )

    return SingleItemResponse[EventRead](
        message="Event updated successfully", data=event
    )


@router.get("", response_model=PaginatedResponse[EventRead])
async def read_my_events_filtered(
    status: Optional[str] = Query(
        None,
        pattern="^(ongoing|past|upcoming)$",
        description="Filter events: ongoing, past (ended), upcoming",
    ),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    now = datetime.now(timezone.utc)

    # Base query: events created by user OR joined
    query = (
        select(Event)
        .join(EventParticipant, Event.id == EventParticipant.event_id, isouter=True)
        .where(
            (Event.created_by_id == current_user.id)
            | (EventParticipant.user_id == current_user.id)
        )
        .distinct()
    )

    # Apply status filter
    if status == "past":
        query = query.filter(Event.end_time < now)
    elif status == "upcoming":
        query = query.filter(Event.start_time > now)
    elif status == "ongoing":
        query = query.filter(Event.start_time <= now, Event.end_time > now)

    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    offset = (page - 1) * per_page

    events = session.exec(
        query.order_by(Event.start_time.desc()).offset(offset).limit(per_page)
    ).all()

    events_with_media = []
    for event in events:
        event_dict = event.model_dump()
        event_dict["cover_photo"] = event.get_cover_photo_media(session)
        events_with_media.append(EventRead(**event_dict))

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[EventRead](
        message=f"User {status or 'all'} events retrieved successfully",
        data=events_with_media,
        pagination=Pagination(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        ),
    )


class ParticipantsResponse(SQLModel):
    data: list[ParticipantRead]
    pagination: Pagination
    stats: dict


@router.get("/{event_id}/participants", response_model=ParticipantsResponse)
async def get_event_participants(
    event_id: int,
    status: Optional[str] = Query(
        None,
        regex="^(approved|pending|rejected)$",
        description="Filter by participant status",
    ),
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
    current_user: User = Depends(get_current_user),
):
    # Check if event exists
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Check if user is event creator or approved participant
    if event.created_by_id != current_user.id:
        participant_check = session.exec(
            select(EventParticipant).where(
                EventParticipant.event_id == event_id,
                EventParticipant.user_id == current_user.id,
            )
        ).first()

        if not participant_check:
            raise HTTPException(
                status_code=403, detail="You are not a participant of this event"
            )
        elif participant_check.status == "pending":
            raise HTTPException(
                status_code=403,
                detail="Your participation request is still pending approval",
            )
        elif participant_check.status == "rejected":
            raise HTTPException(
                status_code=403, detail="Your participation request was rejected"
            )
        elif participant_check.status != "approved":
            raise HTTPException(status_code=403, detail="Access denied")
    # Query EventParticipant table for participants of the event
    statement = select(EventParticipant).where(EventParticipant.event_id == event_id)
    if status:
        statement = statement.where(EventParticipant.status == status)

    total = session.exec(select(func.count()).select_from(statement.subquery())).one()
    offset = (page - 1) * per_page

    event_participants = session.exec(statement.offset(offset).limit(per_page)).all()

    participant_reads = []
    for ep in event_participants:
        user = session.get(User, ep.user_id)
        if user:
            participant_reads.append(
                ParticipantRead(
                    id=user.id,
                    full_name=getattr(user, "full_name", user.username),
                    username=user.username,
                    profile_picture=user.get_profile_picture_media(session),
                    email=user.email,
                    status=ep.status,
                    created_at=ep.created_at,
                )
            )

    # Get stats
    num_total = session.exec(
        select(func.count()).where(EventParticipant.event_id == event_id)
    ).one()
    num_approved = session.exec(
        select(func.count()).where(
            EventParticipant.event_id == event_id, EventParticipant.status == "approved"
        )
    ).one()
    num_pending = session.exec(
        select(func.count()).where(
            EventParticipant.event_id == event_id, EventParticipant.status == "pending"
        )
    ).one()
    num_rejected = session.exec(
        select(func.count()).where(
            EventParticipant.event_id == event_id, EventParticipant.status == "rejected"
        )
    ).one()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return {
        "message": "Event participants retrieved successfully",
        "data": participant_reads,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
        },
        "stats": {
            "num_total": num_total,
            "num_approved": num_approved,
            "num_pending": num_pending,
            "num_rejected": num_rejected,
        },
    }


@router.get("/me", response_model=PaginatedResponse[EventRead])
async def read_my_events(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    total = session.exec(
        select(func.count(Event.id)).where(Event.created_by_id == current_user.id)
    ).one()
    offset = (page - 1) * per_page

    my_events = session.exec(
        select(Event)
        .filter(Event.created_by_id == current_user.id)
        .order_by(Event.start_time.desc())
        .offset(offset)
        .limit(per_page)
    ).all()

    # Attach cover photo
    events_with_media = []
    for event in my_events:
        event_dict = event.model_dump()
        event_dict["cover_photo"] = event.get_cover_photo_media(session)
        events_with_media.append(EventRead(**event_dict))

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[dict](
        message="User events retrieved successfully",
        data=events_with_media,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
        ),
    )


@router.get("/{event_id}", response_model=SingleItemResponse[EventRead])
async def get_single_event(
    event_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Event Privacy Guards: Check access based on privacy setting
    if event.privacy == "private":
        # Private events: only creator or approved participants can access
        if event.created_by_id != current_user.id:
            participant_check = session.exec(
                select(EventParticipant).where(
                    EventParticipant.event_id == event_id,
                    EventParticipant.user_id == current_user.id,
                )
            ).first()

            if not participant_check:
                raise HTTPException(
                    status_code=403,
                    detail="You are not a participant of this private event",
                )
            elif participant_check.status == "pending":
                raise HTTPException(
                    status_code=403,
                    detail="Your participation request is still pending approval",
                )
            elif participant_check.status == "rejected":
                raise HTTPException(
                    status_code=403, detail="Your participation request was rejected"
                )
            elif participant_check.status != "approved":
                raise HTTPException(status_code=403, detail="Access denied")
    elif event.privacy == "public":
        # Public events: anyone can view basic info, but detailed access may require participation
        # For now, we allow public access to public events
        pass

    event_dict = event.model_dump()
    event_dict["cover_photo"] = event.get_cover_photo_media(session)

    # Add host info
    host = session.get(User, event.created_by_id)
    if host:
        event_dict["created_by"] = {
            "id": host.id,
            "username": host.username,
            "email": host.email,
            "photo": host.get_profile_picture_media(session),
        }
    else:
        event_dict["created_by"] = None

    return SingleItemResponse[EventRead](
        message="Event retrieved successfully", data=event_dict
    )


@router.delete("/{event_id}", response_model=dict)
async def delete_event(
    event_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """
    Delete an event and all associated data.
    Only the event creator can delete the event.
    """
    # Get the event
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Only event creator can delete the event
    if event.created_by_id != current_user.id:
        raise HTTPException(
            status_code=403, detail="Only the event creator can delete this event"
        )

    try:
        # Import here to avoid circular imports
        from models import MediaUsage, Media, ContentOwnerType
        from utils.media import delete_media_and_file

        # 1. Delete all media associated with this event
        media_usages = session.exec(
            select(MediaUsage).where(
                MediaUsage.owner_type == ContentOwnerType.EVENT,
                MediaUsage.owner_id == event_id,
            )
        ).all()

        for usage in media_usages:
            if usage.media:
                delete_media_and_file(session, usage.media)

        # 2. Delete all event participants
        participants = session.exec(
            select(EventParticipant).where(EventParticipant.event_id == event_id)
        ).all()

        for participant in participants:
            session.delete(participant)

        # 3. Delete the event itself
        session.delete(event)
        session.commit()

        # --- Notify Participants ---
        from socket_io import sio

        # Fetch participant user IDs (except creator if you want)
        participant_ids = session.exec(
            select(EventParticipant.user_id).where(
                EventParticipant.event_id == event.id
            )
        ).all()

        for user_id in participant_ids:
            send_notification.delay(
                user_id=user_id,
                event="event_deleted",
                data={
                    "event_id": event.id,
                    "event_name": event.name,
                    "message": f"The event '{event.name}' has been deleted by host.",
                },
            )

        return {
            "message": "Event deleted successfully",
            "event_id": event_id,
            "deleted_participants": len(participants),
            "deleted_media": len(media_usages),
        }

    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete event: {str(e)}")


@router.post("/{event_id}/participants/{user_id}/status")
async def update_participant_status(
    event_id: int,
    user_id: int,
    status: str = Body(..., embed=True, regex="^(approved|rejected|pending)$"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Only event creator can update status
    if event.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Prevent self-approval
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot approve/reject your own participation request",
        )

    statement = select(EventParticipant).where(
        EventParticipant.event_id == event_id,
        EventParticipant.user_id == user_id,
    )
    participant = session.exec(statement).first()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    participant.status = status
    session.add(participant)
    session.commit()
    session.refresh(participant)

    # --- Notify the participant ---
    send_notification.delay(
        user_id=user_id,
        event="participant_status_changed",
        data={
            "event_id": event.id,
            "event_name": event.name,
            "status": status,
            "message": f"Your participation request for '{event.name}' has been {status}.",
        },
    )

    return {
        "message": f"Participant {user_id} status updated to {status} for event {event_id}",
        "participant": participant,
    }
