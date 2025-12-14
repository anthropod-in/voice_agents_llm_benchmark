"""Configuration for the long context benchmark (~24K tokens)."""
from pathlib import Path

from benchmarks._shared import turns, ToolsSchemaForTest
from .prompts.system import system_instruction


class BenchmarkConfig:
    """Configuration for the long context benchmark."""

    # Benchmark metadata
    name = "aiwf_long_context"
    description = "Long context benchmark with ~24K token knowledge base"

    # Shared data
    turns = turns
    tools_schema = ToolsSchemaForTest

    # Audio directory path
    audio_dir = Path(__file__).parent.parent / "_shared" / "audio"

    # System prompt
    system_instruction = system_instruction

    @classmethod
    def get_audio_path(cls, turn_index: int) -> Path:
        """Get the audio file path for a specific turn."""
        return cls.audio_dir / f"turn_{turn_index:03d}.wav"
