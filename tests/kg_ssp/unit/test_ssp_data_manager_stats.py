import pytest

from quarl.utils.ssp_data_manager import SSPDataManager


def _local_data_manager():
    return SSPDataManager.__ray_metadata__.modified_class()


@pytest.mark.unit
def test_extraction_rates_use_direct_funnel_numerators() -> None:
    manager = _local_data_manager()
    manager.record_extraction_stats(
        trajectories_count=10,
        answer_matches_count=8,
        valid_questions_count=6,
        successful_problems_count=4,
        format_error_count=4,
    )
    first = manager.get_statistics()
    assert first["current_step_answer_match_rate"] == 0.8
    assert first["current_step_valid_question_rate"] == 0.6
    assert first["current_step_success_rate"] == 0.4

    manager.record_extraction_stats(
        trajectories_count=5,
        answer_matches_count=5,
        valid_questions_count=4,
        successful_problems_count=3,
        format_error_count=1,
    )
    combined = manager.get_statistics()
    assert combined["total_trajectories_processed"] == 15
    assert combined["total_answer_matches"] == 13
    assert combined["total_valid_questions"] == 10
    assert combined["total_successful_problems"] == 7
    assert combined["overall_success_rate"] == pytest.approx(7 / 15)


@pytest.mark.unit
def test_pool_additions_do_not_change_extraction_success_rate() -> None:
    manager = _local_data_manager()
    manager.record_extraction_stats(4, 3, 2, 1, 2)
    manager.add_generated_problems([{} for _ in range(10)], generation_step=1)
    stats = manager.get_statistics()
    assert stats["total_problems_generated"] == 10
    assert stats["overall_success_rate"] == 0.25


@pytest.mark.unit
def test_extraction_funnel_rejects_inconsistent_counts() -> None:
    manager = _local_data_manager()
    with pytest.raises(ValueError, match="successful <= valid <= matched <= trajectories"):
        manager.record_extraction_stats(2, 1, 2, 0, 1)
