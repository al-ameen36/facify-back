from datetime import timezone
from datetime import datetime
from typing import Optional
from fastapi import Depends, APIRouter, HTTPException, Query
from models.events import EventRead
from models.media import Media, MediaUsage
from sqlmodel import Session, func, select
from models import (
    Event,
    EventCreate,
    User,
    UserRead,
    PaginatedResponse,
    SingleItemResponse,
    JoinEventRequest,
    Pagination,
    EventParticipant,
)
from utils.events import create_event, get_event_by_name, get_event_by_name
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
        event = create_event(
            session=session,
            created_by_id=current_user.id,
            name=event_data.name,
            location=event_data.location,
            description=event_data.description,
            start_time=event_data.start_time,
            end_time=event_data.end_time,
            privacy=event_data.privacy,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SingleItemResponse[Event](message="Event created successfully", data=event)


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

    participant = EventParticipant(event_id=event_id, user_id=current_user.id)
    session.add(participant)
    session.commit()

    return {"message": f"{current_user.username} joined the event"}


@router.get("/{event_id}/participants", response_model=PaginatedResponse[UserRead])
async def get_event_participants(
    event_id: int,
    session: Session = Depends(get_session),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1),
    current_user: User = Depends(get_current_user),
):
    # count total participants
    total = session.exec(
        select(func.count(User.id)).join(Event.participants).where(Event.id == event_id)
    ).one()

    offset = (page - 1) * per_page

    participants = session.exec(
        select(User)
        .join(Event.participants)
        .where(Event.id == event_id)
        .offset(offset)
        .limit(per_page)
    ).all()

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[UserRead](
        message="Event participants retrieved successfully",
        data=[UserRead.model_validate(participant) for participant in participants],
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

    cover_photo = event.get_cover_photo(session)
    event_dict = event.dict()
    event_dict["cover_photo"] = cover_photo.url if cover_photo else None

    return SingleItemResponse[EventRead](
        message="Event retrieved successfully", data=EventRead(**event_dict)
    )
