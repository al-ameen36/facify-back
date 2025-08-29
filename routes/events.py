from typing import Annotated
from fastapi import Depends, APIRouter, HTTPException, Query
from sqlmodel import Session, func, select
from models import Event, EventCreate, User, PaginatedResponse, SingleItemResponse
from models.core import Pagination
from utils.events import create_event, get_event_by_name, get_event_by_name
from utils.users import get_current_user
from db import get_session

router = APIRouter(prefix="/events", tags=["event"])


@router.post("/", response_model=SingleItemResponse[Event])
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
            cover_photo=event_data.cover_photo,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SingleItemResponse[Event](message="Event created successfully", data=event)


@router.get("/me", response_model=PaginatedResponse[Event])
async def read_my_events(
    current_user: Annotated[User, Depends(get_current_user)],
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

    total_pages = ((total - 1) // per_page) + 1 if total else 0

    return PaginatedResponse[Event](
        message="User events retrieved successfully",
        data=my_events,
        pagination=Pagination(
            total=total, page=page, per_page=per_page, total_pages=total_pages
        ),
    )
