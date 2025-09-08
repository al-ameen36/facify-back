from datetime import timezone
from datetime import datetime
from typing import Optional
from fastapi import Depends, APIRouter, HTTPException, Query, Body
from models.events import EventRead, ParticipantRead
from sqlmodel import Session, func, select, SQLModel
from models import (
    Event,
    EventCreate,
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

router = APIRouter(prefix="/events", tags=["event"])


@router.post("", response_model=SingleItemResponse[EventRead])
async def add_event(
    event_data: EventCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if get_event_by_name(session, event_data.name):
        raise HTTPException(status_code=400, detail="Event already exists")

    try:
        event = EventCreate(
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
        event = update_event(session=session, event=event, update_data=event_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SingleItemResponse[Event](message="Event updated successfully", data=event)


@router.get("", response_model=PaginatedResponse[EventRead])
async def read_my_events_filtered(
    status: Optional[str] = Query(
        None, regex="^(past|upcoming)$", description="Filter events: past or upcoming"
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
    )

    # Apply status filter
    if status == "past":
        query = query.filter(Event.end_time < now)
    elif status == "upcoming":
        query = query.filter(Event.start_time >= now)

    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    offset = (page - 1) * per_page

    events = session.exec(
        query.order_by(Event.start_time.desc()).offset(offset).limit(per_page)
    ).all()

    # Attach cover photo
    events_with_media = []
    for event in events:
        cover_photo = event.get_cover_photo(session)
        event_dict = event.dict()
        event_dict["cover_photo"] = cover_photo.url if cover_photo else None
        events_with_media.append(EventRead(**event_dict))

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[EventRead](
        message=f"User {status or 'all'} events retrieved successfully",
        data=events_with_media,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
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
                    photo=user.get_profile_picture(session).url,
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


@router.get("/{event_id}", response_model=SingleItemResponse[EventRead])
async def get_single_event(
    event_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    cover_photo = event.get_cover_photo(session)
    event_dict = event.dict()
    event_dict["cover_photo"] = cover_photo.url if cover_photo else None

    # Add host info
    host = session.get(User, event.created_by_id)
    if host:
        event_dict["created_by"] = {
            "id": host.id,
            "username": host.username,
            "email": host.email,
        }
    else:
        event_dict["created_by"] = None

    return SingleItemResponse[EventRead](
        message="Event retrieved successfully", data=event_dict
    )


@router.get("/me", response_model=PaginatedResponse[EventRead])
async def read_my_events(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
):
    total = session.exec(select(func.count(Event.id))).one()
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
        cover_photo = event.get_cover_photo(session)
        event_dict = event.dict()
        event_dict["cover_photo"] = cover_photo.url if cover_photo else None
        events_with_media.append(event_dict)

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[dict](
        message="User events retrieved successfully",
        data=events_with_media,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
        ),
    )


@router.post("/{event_id}/join")
def join_event(
    event_id: int,
    body: JoinEventRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    event = session.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    if event.secret and event.secret != body.secret:
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
    participant = EventParticipant(
        event_id=event_id, user_id=current_user.id, status="pending"
    )
    session.add(participant)
    session.commit()

    return {"message": f"{current_user.username} joined the event (pending approval)"}


@router.post("/{event_id}/participants/{user_id}/status")
def update_participant_status(
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

    return {
        "message": f"Participant {user_id} status updated to {status} for event {event_id}",
        "participant": participant,
    }
