import pytest

from kg_platform.domain.errors import ErrorCode
from kg_platform.domain.models import ToolResult
from kg_platform.domain.tool_outcomes import (
    ToolOutcome,
    ToolOutcomeCode,
    ToolOutcomeStatus,
    classify_tool_result,
    invalid_tool_call,
    render_information,
    system_error,
)


def result(*, text="visible text", count=0, error_code=None):
    return ToolResult(
        operation="get_relations",
        text=text,
        returned_count=count,
        error_code=error_code,
    )


@pytest.mark.parametrize(
    ("tool_result", "status", "code", "visible_text"),
    [
        (result(count=2), ToolOutcomeStatus.SUCCESS, ToolOutcomeCode.OK, "visible text"),
        (
            result(text="No relations found."),
            ToolOutcomeStatus.SUCCESS,
            ToolOutcomeCode.EMPTY_RESULT,
            "No relations found.",
        ),
        (
            result(text="Entity not found.", error_code=ErrorCode.ENTITY_NOT_FOUND.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.INVALID_ENTITY,
            "Entity not found.",
        ),
        (
            result(text="Entity is ambiguous.", error_code=ErrorCode.ENTITY_AMBIGUOUS.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.INVALID_ENTITY,
            "Entity is ambiguous.",
        ),
        (
            result(text="generic", error_code=ErrorCode.RELATION_NOT_FOUND.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.INVALID_RELATION,
            "Relation not found in the active graph.",
        ),
        (
            result(text="latest relation list", error_code=ErrorCode.STALE_RELATION_CHOICE.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.INVALID_RELATION,
            "latest relation list",
        ),
        (
            result(text="not applicable", error_code=ErrorCode.RELATION_NOT_APPLICABLE.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.RELATION_NOT_APPLICABLE,
            "not applicable",
        ),
        (
            result(text="generic", error_code=ErrorCode.INVALID_ARGUMENT.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.INVALID_TOOL_CALL,
            "Error: invalid tool call. Check the tool name and arguments.",
        ),
        (
            result(text="temporarily unavailable", error_code=ErrorCode.BACKEND_UNAVAILABLE.value),
            ToolOutcomeStatus.ERROR,
            ToolOutcomeCode.SYSTEM_ERROR,
            "temporarily unavailable",
        ),
    ],
)
def test_classify_tool_result(tool_result, status, code, visible_text):
    outcome = classify_tool_result(tool_result)

    assert outcome.status is status
    assert outcome.code is code
    assert outcome.text == visible_text
    assert outcome.platform_error_code == tool_result.error_code


def test_factories_and_renderer_hide_structured_envelope_and_exception_details():
    malformed = invalid_tool_call("missing arguments")
    failed = system_error()

    assert render_information(malformed) == (
        "<information>\nError: invalid tool call. missing arguments.\n</information>"
    )
    assert render_information(failed) == (
        "<information>\nThe knowledge graph request failed. "
        "Please try the tool call again.\n</information>"
    )
    assert '"result"' not in render_information(malformed)


def test_outcome_invariants_reject_mismatched_status_and_code():
    with pytest.raises(ValueError, match="SUCCESS outcomes"):
        ToolOutcome(ToolOutcomeStatus.SUCCESS, ToolOutcomeCode.SYSTEM_ERROR, "x")
    with pytest.raises(ValueError, match="ERROR outcomes"):
        ToolOutcome(ToolOutcomeStatus.ERROR, ToolOutcomeCode.OK, "x")
