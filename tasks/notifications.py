import asyncio
from tasks.core import app
from utils.socket import notify_user


@app.task(bind=True, max_retries=3)
async def send_ws_notification_task(self, user_id: int, message: dict):
    try:
        await notify_user(user_id, message)
    except Exception as e:
        self.retry(exc=e, countdown=5)
