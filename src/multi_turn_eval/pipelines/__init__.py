"""Pipeline implementations for different LLM service types.

Pipelines handle the full execution of multi-turn benchmarks including:
- Creating and configuring LLM services
- Managing turn flow (queuing turns, detecting end-of-turn)
- Recording transcripts and metrics
- Handling reconnection for long-running sessions

Available pipelines:
- TextPipeline: For text-based LLM services (OpenAI, Anthropic, Google, etc.)
- RealtimePipeline: For speech-to-speech services (OpenAI Realtime, Gemini Live)
- NovaSonicPipeline: For AWS Nova Sonic speech-to-speech service
"""

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.pipelines.text import TextPipeline
from multi_turn_eval.pipelines.realtime import (
    RealtimePipeline,
    GeminiLiveLLMServiceWithReconnection,
)
from multi_turn_eval.pipelines.nova_sonic import (
    NovaSonicPipeline,
    NovaSonicLLMServiceWithCompletionSignal,
    NovaSonicTurnEndDetector,
)

__all__ = [
    "BasePipeline",
    "TextPipeline",
    "RealtimePipeline",
    "GeminiLiveLLMServiceWithReconnection",
    "NovaSonicPipeline",
    "NovaSonicLLMServiceWithCompletionSignal",
    "NovaSonicTurnEndDetector",
]
