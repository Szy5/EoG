from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    ENTITY_NOT_FOUND = "ENTITY_NOT_FOUND"
    RELATION_NOT_FOUND = "RELATION_NOT_FOUND"
    ENTITY_AMBIGUOUS = "ENTITY_AMBIGUOUS"
    RELATION_NOT_APPLICABLE = "RELATION_NOT_APPLICABLE"
    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"
    RESULT_LIMIT_EXCEEDED = "RESULT_LIMIT_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_CONTEXT_MISMATCH = "SESSION_CONTEXT_MISMATCH"
    STALE_RELATION_CHOICE = "STALE_RELATION_CHOICE"


RETRYABLE_CODES = {
    ErrorCode.RATE_LIMITED,
    ErrorCode.BACKEND_UNAVAILABLE,
    ErrorCode.DEADLINE_EXCEEDED,
}


class KGPlatformError(Exception):
    def __init__(self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = code in RETRYABLE_CODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
