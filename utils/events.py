from typing import Optional
from sqlmodel import Session, select
from dotenv import load_dotenv
from models import Event, EventCreate

load_dotenv()


def get_event_by_name(session: Session, name: str) -> Optional[Event]:
    return session.exec(select(Event).where(Event.name == name)).first()


def create_event(session: Session, event: EventCreate) -> Event:
    event = EventCreate.model_validate(event)
    session.add(event)
    session.commit()
    session.refresh(event)
    return event


def update_event(session: Session, event: Event, update_data: EventCreate) -> Event:
    for field, value in update_data.model_dump(exclude_unset=True).items():
        setattr(event, field, value)

    session.add(event)
    session.commit()
    session.refresh(event)

    return event
