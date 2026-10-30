from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import ErrorCode
from .models import ToolResult


class ToolOutcomeStatus(str, Enum):
    """Broad result class used for reporting every attempted KG action."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


class ToolOutcomeCode(str, Enum):
    """Stable model-independent categories for KG tool outcome statistics."""

    OK = "OK"
    EMPTY_RESULT = "EMPTY_RESULT"
    INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
    INVALID_ENTITY = "INVALID_ENTITY"
    INVALID_RELATION = "INVALID_RELATION"
    RELATION_NOT_APPLICABLE = "RELATION_NOT_APPLICABLE"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass(frozen=True)
class ToolOutcome:
    """Normalized result whose text is the only content shown to the solver."""

    status: ToolOutcomeStatus
    code: ToolOutcomeCode
    text: str
    platform_error_code: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("ToolOutcome text must be non-empty")
        if self.status is ToolOutcomeStatus.SUCCESS and self.code not in {
            ToolOutcomeCode.OK,
            ToolOutcomeCode.EMPTY_RESULT,
        }:
            raise ValueError("SUCCESS outcomes require OK or EMPTY_RESULT")
        if self.status is ToolOutcomeStatus.ERROR and self.code in {
            ToolOutcomeCode.OK,
            ToolOutcomeCode.EMPTY_RESULT,
        }:
            raise ValueError("ERROR outcomes require an error code")


_ENTITY_ERRORS = {
    ErrorCode.ENTITY_NOT_FOUND.value,
    ErrorCode.ENTITY_AMBIGUOUS.value,
}
_RELATION_ERRORS = {
    ErrorCode.RELATION_NOT_FOUND.value,
    ErrorCode.STALE_RELATION_CHOICE.value,
}


def classify_tool_result(result: ToolResult) -> ToolOutcome:
    """Collapse detailed KG Platform errors into the public reporting taxonomy."""

    if result.error_code is None:
        code = ToolOutcomeCode.EMPTY_RESULT if result.returned_count == 0 else ToolOutcomeCode.OK
        return ToolOutcome(ToolOutcomeStatus.SUCCESS, code, result.text)

    text = result.text
    if result.error_code in _ENTITY_ERRORS:
        code = ToolOutcomeCode.INVALID_ENTITY
    elif result.error_code in _RELATION_ERRORS:
        code = ToolOutcomeCode.INVALID_RELATION
        if result.error_code == ErrorCode.RELATION_NOT_FOUND.value:
            text = "Relation not found in the active graph."
    elif result.error_code == ErrorCode.RELATION_NOT_APPLICABLE.value:
        code = ToolOutcomeCode.RELATION_NOT_APPLICABLE
    elif result.error_code == ErrorCode.INVALID_ARGUMENT.value:
        code = ToolOutcomeCode.INVALID_TOOL_CALL
        text = "Error: invalid tool call. Check the tool name and arguments."
    else:
        code = ToolOutcomeCode.SYSTEM_ERROR
    return ToolOutcome(ToolOutcomeStatus.ERROR, code, text, result.error_code)


def invalid_tool_call(detail: str) -> ToolOutcome:
    detail = detail.strip().rstrip(".")
    text = f"Error: invalid tool call. {detail}." if detail else "Error: invalid tool call."
    return ToolOutcome(ToolOutcomeStatus.ERROR, ToolOutcomeCode.INVALID_TOOL_CALL, text)


def system_error() -> ToolOutcome:
    return ToolOutcome(
        ToolOutcomeStatus.ERROR,
        ToolOutcomeCode.SYSTEM_ERROR,
        "The knowledge graph request failed. Please try the tool call again.",
    )


def render_information(outcome: ToolOutcome) -> str:
    """Render main-branch-compatible plain text instead of exposing result JSON."""

    return f"<information>\n{outcome.text.strip()}\n</information>"
