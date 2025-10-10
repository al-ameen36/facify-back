from db import engine
from sqlmodel import Session, select
from models import User
from utils.users import decode_token

import socketio

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    cors_credentials=True,
    logger=True,
    engineio_logger=True,
    client_manager=socketio.AsyncRedisManager("redis://localhost:6379/1"),
)


def get_user_from_token(session: Session, token: str) -> User | None:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = session.exec(select(User).where(User.username == username)).first()
    return user


@sio.event
async def connect(sid, environ, auth):
    token = auth.get("token")

    with Session(engine) as session:
        user = get_user_from_token(session, token)

        if not user:
            await sio.disconnect(sid)
            return

        await sio.enter_room(sid, f"user:{user.id}")
        print(f"User {user.id} connected with sid {sid}")


@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")
