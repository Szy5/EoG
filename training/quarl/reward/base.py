import importlib.util
import sys
from functools import partial
from typing import Dict

from quarl.interface import RewardFuncInfo

from .score.kgqa_answer_em import compute_score_batch as reward_kgqa_answer_em_fn
from .score.kgqa_format_reward import compute_score_batch as reward_kgqa_format_reward_fn
from .score.kgqa_protocol_gate import compute_score_batch as reward_kgqa_protocol_gate_fn
from .score.process_score import compute_score_batch as reward_process_score_fn
from .score.outcome_score import compute_score_batch as reward_outcome_score_fn

REWARD_FUNCTIONS = {
    "kgqa_answer_em": reward_kgqa_answer_em_fn,
    "kgqa_format_reward": reward_kgqa_format_reward_fn,
    "kgqa_protocol_gate": reward_kgqa_protocol_gate_fn,
    "process_score": reward_process_score_fn,
    "outcome_score": reward_outcome_score_fn,
}


def get_custom_reward_fns(config) -> Dict[str, RewardFuncInfo]:

    reward_fns_config = config.get("custom_reward_functions", {})
    if not reward_fns_config:
        return {}

    reward_func_dict = {}
    print(f"using customized reward functions: {reward_fns_config}")

    for reward_fn_name, reward_config in reward_fns_config.items():
        if reward_fn_name not in REWARD_FUNCTIONS:
            print(f"[quarl.reward]: Warning: reward_fn `{reward_fn_name}` not exists!")
            continue

        reward_fn = REWARD_FUNCTIONS[reward_fn_name]
        kwargs = reward_config.get("kwargs", {})

        reward_func_dict[reward_fn_name] = RewardFuncInfo(
            name=reward_fn_name,
            reward_fn=partial(reward_fn, **kwargs),
            labels=reward_config.get("labels", []),
            integration=reward_config.get("integration", "sum"),
        )

        print(f"Successfully load reward_fn: {reward_fn_name}")

    return reward_func_dict
