"""Shim for backward compatibility - imports system_instruction from benchmarks."""
# Re-export from the new canonical location (long context is the default)
from benchmarks.aiwf_long_context.prompts.system import system_instruction

__all__ = ["system_instruction"]
