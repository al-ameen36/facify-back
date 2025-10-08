from sqlmodel import Session
from db import engine
from utils.socket import get_user_from_token
import socketio

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    cors_credentials=True,
    logger=True,
    engineio_logger=True,
    client_manager=socketio.AsyncRedisManager("redis://localhost:6379/1"),
)


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
