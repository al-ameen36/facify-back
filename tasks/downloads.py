import os
import zipfile
import requests
from pathlib import Path
from datetime import datetime, timezone
from sqlmodel import select
from models import Media
from models.downloads import DownloadJob, DownloadJobStatus
from db import get_session
from workers import app
from tasks.notifications import send_notification

DOWNLOAD_DIR = Path("/tmp/downloads")


@app.task(name="tasks.downloads.generate_bulk_download", bind=True, max_retries=3)
def generate_bulk_download(self, job_id: str):
    """
    Celery task to generate ZIP file for a download job.
    Takes only job_id, loads everything from the database.
    """
    with next(get_session()) as session:
        try:
            # Load the job
            job = session.get(DownloadJob, job_id)
            if not job:
                print(f"Job {job_id} not found")
                return

            # Update status to processing
            job.status = DownloadJobStatus.PROCESSING
            session.add(job)
            session.commit()

            # Ensure download directory exists
            DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

            # Create ZIP file
            zip_path = DOWNLOAD_DIR / f"{job_id}.zip"

            successful_downloads = 0
            failed_downloads = 0

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for media_id in job.media_ids:
                    # Load media from database
                    media = session.get(Media, media_id)
                    if not media or not media.external_url:
                        print(f"Media {media_id} not found or has no external_url")
                        failed_downloads += 1
                        continue

                    try:
                        # Download file from ImageKit
                        print(f"Downloading media {media_id} from {media.external_url}")
                        response = requests.get(
                            media.external_url, timeout=60, stream=True
                        )
                        response.raise_for_status()

                        # Use filename from media record
                        filename = media.filename or f"media_{media_id}.jpg"

                        # Handle duplicate filenames in ZIP
                        base_name, ext = os.path.splitext(filename)
                        counter = 1
                        final_filename = filename

                        while final_filename in zipf.namelist():
                            final_filename = f"{base_name}_{counter}{ext}"
                            counter += 1

                        # Write to ZIP
                        zipf.writestr(final_filename, response.content)
                        successful_downloads += 1
                        print(f"Successfully added {final_filename} to ZIP")

                    except Exception as e:
                        print(f"Error downloading media {media_id}: {e}")
                        failed_downloads += 1
                        continue

            # Check if we have any successful downloads
            if successful_downloads == 0:
                raise Exception("Failed to download any media files")

            # Update job as completed
            job.status = DownloadJobStatus.COMPLETED
            job.file_path = str(zip_path)
            if failed_downloads > 0:
                job.error_message = f"{failed_downloads} of {len(job.media_ids)} files failed to download"
            session.add(job)
            session.commit()

            # Notify user that download is ready
            send_notification.delay(
                user_id=job.user_id,
                event="download_ready",
                data={
                    "job_id": job.id,
                    "file_name": f"download_{job.id}.zip",
                },
            )

            print(
                f"Download job {job_id} completed: {successful_downloads} succeeded, {failed_downloads} failed"
            )

        except Exception as e:
            print(f"Task failed: {e}")

            # Clean up partial ZIP if exists
            zip_path = DOWNLOAD_DIR / f"{job_id}.zip"
            if zip_path.exists():
                try:
                    os.remove(zip_path)
                except Exception:
                    pass

            # Update job status on failure
            job = session.get(DownloadJob, job_id)
            if job:
                job.status = DownloadJobStatus.FAILED
                job.error_message = str(e)
                session.add(job)
                session.commit()

            # Retry with 20 second countdown
            self.retry(exc=e, countdown=20)


@app.task(name="tasks.downloads.cleanup_expired_downloads")
def cleanup_expired_downloads():
    """
    Periodic task to clean up expired download jobs.
    Add to Celery beat schedule in workers.py:

    from celery.schedules import crontab

    app.conf.beat_schedule = {
        'cleanup-expired-downloads': {
            'task': 'tasks.downloads.cleanup_expired_downloads',
            'schedule': crontab(hour='*'),  # Run every hour
        },
    }
    """
    with next(get_session()) as session:
        try:
            # Find expired jobs
            now = datetime.now(timezone.utc)

            # Get all jobs that might be expired
            all_jobs = session.exec(
                select(DownloadJob).where(
                    DownloadJob.status.in_(
                        [
                            DownloadJobStatus.COMPLETED,
                            DownloadJobStatus.PROCESSING,
                            DownloadJobStatus.PENDING,
                        ]
                    )
                )
            ).all()

            expired_jobs = []
            for job in all_jobs:
                if job.expires_at:
                    expires_at = (
                        job.expires_at.replace(tzinfo=timezone.utc)
                        if job.expires_at.tzinfo is None
                        else job.expires_at
                    )
                    if expires_at < now:
                        expired_jobs.append(job)

            cleaned = 0
            for job in expired_jobs:
                # Delete file if exists
                if job.file_path and os.path.exists(job.file_path):
                    try:
                        os.remove(job.file_path)
                        cleaned += 1
                    except Exception as e:
                        print(f"Error deleting file {job.file_path}: {e}")

                # Update job status
                job.status = DownloadJobStatus.EXPIRED
                job.file_path = None
                session.add(job)

            session.commit()
            print(f"Cleaned up {cleaned} expired download files")

        except Exception as e:
            print(f"Error in cleanup task: {e}")
            session.rollback()
