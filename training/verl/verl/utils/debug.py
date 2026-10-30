"""Compatibility wrappers for older veRL imports.

The local veRL snapshot in this repository expects ``verl.utils.debug``,
while newer code moved these utilities under ``verl.utils.profiler``.
"""

from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, marked_timer, simple_timer

__all__ = ["GPUMemoryLogger", "log_gpu_memory_usage", "marked_timer", "simple_timer"]
