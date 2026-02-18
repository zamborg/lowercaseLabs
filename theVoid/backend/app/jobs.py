from __future__ import annotations

from datetime import timedelta
import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .db import now_utc
from .llm import extract_insights
from .models import (
    Entry,
    EntryStatus,
    Insight,
    Job,
    JobStatus,
    JobType,
    SocialPresence,
    Transcript,
)
from .social import mood_to_dot_color
from .storage import LocalObjectStorage
from .transcription import transcribe_entry

logger = logging.getLogger(__name__)


def enqueue_job(
    db: Session,
    job_type: JobType,
    payload: dict,
    trace_id: str,
    max_attempts: int = 3,
    delay_seconds: int = 0,
) -> Job:
    job = Job(
        job_type=job_type,
        payload=payload,
        trace_id=trace_id,
        status=JobStatus.PENDING,
        max_attempts=max_attempts,
        run_after=now_utc() + timedelta(seconds=delay_seconds),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def claim_next_job(db: Session) -> Job | None:
    now = now_utc()
    job = (
        db.query(Job)
        .filter(Job.status == JobStatus.PENDING, Job.run_after <= now)
        .order_by(Job.created_at.asc())
        .first()
    )
    if job is None:
        return None

    job.status = JobStatus.RUNNING
    job.locked_at = now
    job.attempt_count += 1
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def mark_job_succeeded(db: Session, job: Job) -> None:
    job.status = JobStatus.SUCCEEDED
    job.finished_at = now_utc()
    job.last_error = None
    db.add(job)
    db.commit()


def mark_job_failed(db: Session, job: Job, error: str) -> None:
    job.last_error = error
    if job.attempt_count < job.max_attempts:
        job.status = JobStatus.PENDING
        # Basic linear backoff.
        job.run_after = now_utc() + timedelta(seconds=job.attempt_count * 15)
        job.locked_at = None
    else:
        job.status = JobStatus.FAILED
        job.finished_at = now_utc()
    db.add(job)
    db.commit()


def process_transcription_job(db: Session, storage: LocalObjectStorage, payload: dict, trace_id: str) -> None:
    entry_id = payload["entry_id"]
    entry = db.get(Entry, entry_id)
    if entry is None:
        logger.warning("missing_entry_for_transcription", extra={"trace_id": trace_id, "entry_id": entry_id})
        return

    entry.status = EntryStatus.TRANSCRIBING
    db.add(entry)
    db.commit()

    transcript_text, metadata = transcribe_entry(entry, storage)

    transcript = db.get(Transcript, entry.id)
    if transcript is None:
        transcript = Transcript(entry_id=entry.id, text=transcript_text, provider_metadata=metadata)
    else:
        transcript.text = transcript_text
        transcript.provider_metadata = metadata

    db.add(transcript)
    db.commit()

    enqueue_job(
        db,
        job_type=JobType.INSIGHTS,
        payload={"entry_id": entry.id, "user_id": entry.user_id},
        trace_id=trace_id,
    )


def process_insights_job(db: Session, payload: dict, trace_id: str) -> None:
    entry_id = payload["entry_id"]
    entry = db.get(Entry, entry_id)
    if entry is None:
        logger.warning("missing_entry_for_insights", extra={"trace_id": trace_id, "entry_id": entry_id})
        return

    entry.status = EntryStatus.ANALYZING
    db.add(entry)
    db.commit()

    transcript = db.get(Transcript, entry.id)
    if transcript is None:
        raise RuntimeError(f"Transcript missing for entry {entry.id}")

    extracted = extract_insights(transcript.text)

    insight = db.get(Insight, entry.id)
    if insight is None:
        insight = Insight(entry_id=entry.id, **extracted)
    else:
        insight.mood_score = extracted["mood_score"]
        insight.mood_tags = extracted["mood_tags"]
        insight.summary = extracted["summary"]
        insight.themes = extracted["themes"]
        insight.signals = extracted["signals"]
        insight.safety_flags = extracted["safety_flags"]

    db.add(insight)

    entry.status = EntryStatus.READY
    db.add(entry)
    db.commit()

    enqueue_job(
        db,
        job_type=JobType.SOCIAL_AGGREGATION,
        payload={"entry_id": entry.id, "user_id": entry.user_id, "local_date": entry.local_date.isoformat()},
        trace_id=trace_id,
    )


def process_social_aggregation_job(db: Session, payload: dict, trace_id: str) -> None:
    entry_id = payload["entry_id"]
    entry = db.get(Entry, entry_id)
    if entry is None:
        logger.warning("missing_entry_for_social", extra={"trace_id": trace_id, "entry_id": entry_id})
        return

    insight = db.get(Insight, entry.id)
    if insight is None:
        logger.warning("missing_insight_for_social", extra={"trace_id": trace_id, "entry_id": entry_id})
        return

    color = mood_to_dot_color(insight.mood_score)

    presence = (
        db.query(SocialPresence)
        .filter(SocialPresence.user_id == entry.user_id, SocialPresence.local_date == entry.local_date)
        .one_or_none()
    )
    if presence is None:
        presence = SocialPresence(
            user_id=entry.user_id,
            local_date=entry.local_date,
            dot_color=color,
        )
    else:
        presence.dot_color = color

    db.add(presence)
    try:
        db.commit()
        return
    except IntegrityError:
        db.rollback()

    # Handle concurrent insert race for (user_id, local_date).
    existing = (
        db.query(SocialPresence)
        .filter(SocialPresence.user_id == entry.user_id, SocialPresence.local_date == entry.local_date)
        .one_or_none()
    )
    if existing is None:
        raise
    existing.dot_color = color
    db.add(existing)
    db.commit()


def process_job(db: Session, storage: LocalObjectStorage, job: Job) -> None:
    if job.job_type == JobType.TRANSCRIPTION:
        process_transcription_job(db, storage, job.payload, job.trace_id)
        return

    if job.job_type == JobType.INSIGHTS:
        process_insights_job(db, job.payload, job.trace_id)
        return

    if job.job_type == JobType.SOCIAL_AGGREGATION:
        process_social_aggregation_job(db, job.payload, job.trace_id)
        return

    raise RuntimeError(f"Unsupported job type: {job.job_type}")
