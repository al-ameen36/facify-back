from sqlmodel import Session, select
from models import User
from utils.users import decode_token


def get_user_from_token(session: Session, token: str) -> User | None:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = session.exec(select(User).where(User.username == username)).first()
    return user


async def notify_user(user_id: int, message: dict):
    from socket_io import sio

    await sio.emit("notification", message, room=f"user:{user_id}")
