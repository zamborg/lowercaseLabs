from __future__ import annotations

import logging
import time
from threading import Event

from .config import settings
from .db import Base, SessionLocal, engine
from .jobs import claim_next_job, mark_job_failed, mark_job_succeeded, process_job
from .storage import LocalObjectStorage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("thevoid.worker")


def run_forever(stop_event: Event | None = None) -> None:
    Base.metadata.create_all(bind=engine)
    storage = LocalObjectStorage(settings.object_storage_root, settings.jwt_secret)

    logger.info("worker_started")
    while stop_event is None or not stop_event.is_set():
        db = SessionLocal()
        job = None
        try:
            job = claim_next_job(db)
            if job is None:
                time.sleep(settings.worker_poll_interval_seconds)
                continue

            logger.info(
                "job_claimed",
                extra={
                    "job_id": job.id,
                    "job_type": job.job_type.value,
                    "trace_id": job.trace_id,
                    "attempt": job.attempt_count,
                },
            )

            process_job(db, storage, job)
            mark_job_succeeded(db, job)
        except Exception as exc:
            logger.exception("job_failed", extra={"job_id": getattr(job, "id", None)})
            if job is not None:
                mark_job_failed(db, job, str(exc))
            time.sleep(1)
        finally:
            db.close()


def run() -> None:
    run_forever()


if __name__ == "__main__":
    run()
