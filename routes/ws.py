from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlmodel import Session, select
from db import get_session
from models import User
from utils.users import decode_token
from utils.ws import active_connections
from dotenv import load_dotenv
import os

load_dotenv()

router = APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


async def get_user_from_token(token: str, session: Session) -> User | None:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    username = payload.get("sub")
    if not username:
        return None

    user = session.exec(select(User).where(User.username == username)).first()
    return user


@router.websocket("/ws/notifications")
async def websocket_endpoint(
    websocket: WebSocket, session: Session = Depends(get_session)
):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001)
        return

    user = await get_user_from_token(token, session)

    if not user:
        await websocket.close(code=4002)
        return

    await websocket.accept()
    active_connections[user.id] = websocket

    await websocket.send_json({"msg": "Connected"})

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        active_connections.pop(user.id, None)
        print(f"Client disconnected: {user.username}")
