
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "review_audit.sqlite3"

router = APIRouter(tags=["Review and Audit"])


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    study_id: str | None = Field(default=None, max_length=128)
    predicted_label: str | None = Field(default=None, max_length=100)
    probability: float | None = Field(default=None, ge=0, le=1)
    comment: str = Field(default="", max_length=1000)


def initialize_audit_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS review_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                decision TEXT NOT NULL
                    CHECK (decision IN ('approved', 'rejected')),
                study_id TEXT,
                predicted_label TEXT,
                probability REAL,
                comment TEXT NOT NULL
            )
            """
        )


def record_review(review: ReviewDecision) -> dict:
    initialize_audit_db()
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute(
                """
                INSERT INTO review_audit (
                    timestamp_utc,
                    decision,
                    study_id,
                    predicted_label,
                    probability,
                    comment
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    review.decision,
                    review.study_id,
                    review.predicted_label,
                    review.probability,
                    review.comment,
                ),
            )
            review_id = cursor.lastrowid

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not save the review decision.",
        ) from exc

    return {
        "id": review_id,
        "timestamp_utc": timestamp,
        **review.model_dump(),
    }


@router.post("/review-decisions", status_code=201)
def create_review_decision(review: ReviewDecision):
    return record_review(review)


@router.get("/audit-logs")
def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=200),
):
    initialize_audit_db()

    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, timestamp_utc, decision, study_id,
                       predicted_label, probability, comment
                FROM review_audit
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not retrieve the audit history.",
        ) from exc

    return {
        "count": len(rows),
        "items": [dict(row) for row in rows],
    }
