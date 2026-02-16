"""Pipeline implementations for different LLM service types.

Pipelines handle the full execution of multi-turn benchmarks including:
- Creating and configuring LLM services
- Managing turn flow (queuing turns, detecting end-of-turn)
- Recording transcripts and metrics
- Handling reconnection for long-running sessions

Available pipelines:
- TextPipeline: For text-based LLM services (OpenAI, Anthropic, Google, etc.)
- RealtimePipeline: For speech-to-speech services (OpenAI Realtime, Gemini Live)
- GrokRealtimePipeline: For xAI Grok Voice Agent API
- NovaSonicPipeline: For AWS Nova Sonic speech-to-speech service
"""

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.pipelines.text import TextPipeline

# Optional audio/realtime pipelines may depend on heavy audio packages.
# Keep text-only usage working when those extras are not installed.
try:
    from multi_turn_eval.pipelines.realtime import (
        RealtimePipeline,
        GeminiLiveLLMServiceWithReconnection,
    )
except ModuleNotFoundError:
    RealtimePipeline = None
    GeminiLiveLLMServiceWithReconnection = None

try:
    from multi_turn_eval.pipelines.grok_realtime import (
        GrokRealtimePipeline,
        XAIRealtimeLLMService,
    )
except ModuleNotFoundError:
    GrokRealtimePipeline = None
    XAIRealtimeLLMService = None

try:
    from multi_turn_eval.pipelines.nova_sonic import (
        NovaSonicPipeline,
        NovaSonicLLMServiceWithCompletionSignal,
        NovaSonicTurnGate,
    )
except ModuleNotFoundError:
    NovaSonicPipeline = None
    NovaSonicLLMServiceWithCompletionSignal = None
    NovaSonicTurnGate = None

__all__ = [
    "BasePipeline",
    "TextPipeline",
]

if RealtimePipeline is not None:
    __all__.extend(["RealtimePipeline", "GeminiLiveLLMServiceWithReconnection"])
if GrokRealtimePipeline is not None:
    __all__.extend(["GrokRealtimePipeline", "XAIRealtimeLLMService"])
if NovaSonicPipeline is not None:
    __all__.extend(
        ["NovaSonicPipeline", "NovaSonicLLMServiceWithCompletionSignal", "NovaSonicTurnGate"]
    )
