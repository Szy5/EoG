import pytest


try:
    from quarl.worker.rollout.sglang_rollout import should_force_final_answer
    from verl.workers.rollout.schemas import AsyncRolloutRequestStateEnum
except ModuleNotFoundError as exc:
    should_force_final_answer = None
    AsyncRolloutRequestStateEnum = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.mark.unit
@pytest.mark.skipif(_IMPORT_ERROR is not None, reason="optional SGLang rollout dependencies are unavailable")
def test_forced_final_answer_is_validation_only() -> None:
    kwargs = dict(
        enabled=True,
        current_turns=10,
        max_assistant_turns=10,
        state=AsyncRolloutRequestStateEnum.TOOL_CALLING,
        has_answer=False,
    )

    assert should_force_final_answer(is_validate=True, **kwargs)
    assert not should_force_final_answer(is_validate=False, **kwargs)


@pytest.mark.unit
@pytest.mark.skipif(_IMPORT_ERROR is not None, reason="optional SGLang rollout dependencies are unavailable")
def test_forced_final_answer_requires_final_unanswered_tool_call() -> None:
    base = dict(
        is_validate=True,
        enabled=True,
        current_turns=10,
        max_assistant_turns=10,
        state=AsyncRolloutRequestStateEnum.TOOL_CALLING,
        has_answer=False,
    )

    assert not should_force_final_answer(**{**base, "enabled": False})
    assert not should_force_final_answer(**{**base, "current_turns": 9})
    assert not should_force_final_answer(**{**base, "has_answer": True})
    assert not should_force_final_answer(
        **{**base, "state": AsyncRolloutRequestStateEnum.RUNNING}
    )
