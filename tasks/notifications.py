from workers import app, emit_to_user


@app.task(bind=True, max_retries=3)
async def send_notification(self, user_id: int, event: str, data: dict):

    try:
        await emit_to_user(user_id, event, data)
    except Exception as e:
        self.retry(exc=e, countdown=5)
