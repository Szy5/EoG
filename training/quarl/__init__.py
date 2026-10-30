"""Top-level exports for quarl.

Keep imports lazy so unit tests and lightweight utilities do not need to
import the full trainer stack during module collection.
"""

from importlib import import_module
from typing import Any


__all__ = ["SSPRayPPOTrainer", "SSPDataManager", "ProblemExtractor"]


def __getattr__(name: str) -> Any:
    if name == "SSPRayPPOTrainer":
        return import_module("quarl.trainer.ppo.ssp_ray_trainer").SSPRayPPOTrainer
    if name == "SSPDataManager":
        return import_module("quarl.utils.ssp_data_manager").SSPDataManager
    if name == "ProblemExtractor":
        return import_module("quarl.utils.problem_extraction").ProblemExtractor
    raise AttributeError(f"module 'quarl' has no attribute {name!r}")
