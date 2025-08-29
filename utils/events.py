from typing import Optional
from sqlmodel import Session, select
from dotenv import load_dotenv
from models import Event

load_dotenv()


def get_event_by_name(session: Session, name: str) -> Optional[Event]:
    return session.exec(select(Event).where(Event.name == name)).first()


def create_event(
    session: Session,
    name: str,
    description: str,
    cover_photo: str,
    start_time: str,
    end_time: str,
    location: str,
    privacy: str,
    created_by_id: str,
) -> Event:
    event = Event(
        name=name,
        description=description,
        cover_photo=cover_photo,
        start_time=start_time,
        end_time=end_time,
        location=location,
        privacy=privacy,
        created_by_id=created_by_id,
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return event
