import asyncio
from fastapi import WebSocket

active_connections: dict[int, WebSocket] = {}


async def send_ws_message(user_id: int, message: dict):
    ws = active_connections.get(user_id)
    if not ws:
        print(f"⚠️ No active WS connection for user {user_id}")
        return
    try:
        await ws.send_json(message)
    except Exception as e:
        print(f"⚠️ Removing dead WS for {user_id}: {e}")
        active_connections.pop(user_id, None)


def notify_ws(user_id: int, message: dict):
    """Safe to call from sync or async routes, non-blocking."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(send_ws_message(user_id, message))
    except RuntimeError:
        # Not inside an event loop (e.g. running in thread)
        asyncio.run(send_ws_message(user_id, message))
