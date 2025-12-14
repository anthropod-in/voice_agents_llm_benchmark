"""Shim for backward compatibility - imports ToolsSchemaForTest from benchmarks._shared."""
# Re-export from the new canonical location
from benchmarks._shared.tools import ToolsSchemaForTest

__all__ = ["ToolsSchemaForTest"]
