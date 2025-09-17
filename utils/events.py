from typing import Optional
from sqlmodel import Session, select
from dotenv import load_dotenv
from models import Event, EventCreateDB, User

load_dotenv()


def get_event_by_name(session: Session, name: str) -> Optional[Event]:
    return session.exec(select(Event).where(Event.name == name)).first()


def create_event(session: Session, event: EventCreateDB) -> Event:
    event_data = EventCreateDB.model_validate(event)
    db_event = Event(**event_data.model_dump())

    session.add(db_event)
    session.commit()
    session.refresh(db_event)

    return db_event


def update_event(
    session: Session, event: Event, update_data: EventCreateDB, created_by_id: int
) -> Event:
    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(event, field, value)

    event.created_by_id = created_by_id
    session.add(event)
    session.commit()
    session.refresh(event)

    return event
