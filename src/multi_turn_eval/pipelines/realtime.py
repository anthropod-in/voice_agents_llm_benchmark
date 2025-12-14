"""Realtime pipeline for OpenAI Realtime and Gemini Live models.

This pipeline works with speech-to-speech models that use audio input/output:
- OpenAI Realtime (gpt-realtime)
- Gemini Live (gemini-*-native-audio-*)

Pipeline:
    paced_input → context_aggregator.user() → transcript.user() →
    llm → ToolCallRecorder → assistant_shim → context_aggregator.assistant()
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import soundfile as sf
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    MetricsFrame,
    TranscriptionMessage,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.openai.realtime import events as rt_events
from pipecat.transports.base_transport import TransportParams

from multi_turn_eval.pipelines.base import BasePipeline
from multi_turn_eval.processors.tool_call_recorder import ToolCallRecorder
from multi_turn_eval.processors.tts_transcript import TTSStoppedAssistantTranscriptProcessor
from multi_turn_eval.transports.paced_input import PacedInputTransport


class GeminiLiveLLMServiceWithReconnection(GeminiLiveLLMService):
    """Extended Gemini Live service that exposes reconnection events.

    The base GeminiLiveLLMService handles reconnection internally when the
    10-minute session timeout occurs, but doesn't expose events for external
    coordination. This subclass:

    1. Calls on_reconnecting callback before disconnecting
    2. Calls on_reconnected callback after reconnecting
    3. Tracks whether we were in the middle of receiving a response

    This allows the test harness to:
    - Pause audio input during reconnection
    - Re-queue the interrupted turn's audio after reconnection
    - Reset turn tracking state
    """

    def __init__(
        self,
        on_reconnecting: Optional[Callable[[], None]] = None,
        on_reconnected: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """Initialize with optional reconnection callbacks.

        Args:
            on_reconnecting: Called before disconnecting during reconnection.
                            Use this to pause audio input and save state.
            on_reconnected: Called after reconnection completes.
                           Use this to resume audio input and re-queue interrupted turn.
        """
        super().__init__(**kwargs)
        self._on_reconnecting = on_reconnecting
        self._on_reconnected = on_reconnected
        self._reconnecting = False

    def is_reconnecting(self) -> bool:
        """Check if currently in the middle of a reconnection."""
        return self._reconnecting

    async def _reconnect(self):
        """Override to call callbacks before/after reconnection."""
        self._reconnecting = True

        # Call on_reconnecting callback
        if self._on_reconnecting:
            try:
                logger.info("GeminiLiveWithReconnection: Calling on_reconnecting callback")
                self._on_reconnecting()
            except Exception as e:
                logger.warning(f"Error in on_reconnecting callback: {e}")

        # Call parent reconnect implementation
        try:
            await super()._reconnect()
        finally:
            self._reconnecting = False

        # Call on_reconnected callback
        if self._on_reconnected:
            try:
                logger.info("GeminiLiveWithReconnection: Calling on_reconnected callback")
                self._on_reconnected()
            except Exception as e:
                logger.warning(f"Error in on_reconnected callback: {e}")


class LLMFrameLogger(FrameProcessor):
    """Logs every frame emitted by the LLM stage and captures TTFB metrics."""

    def __init__(self, recorder_accessor):
        super().__init__()
        self._recorder_accessor = recorder_accessor

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            logger.debug(f"[LLM→] {frame.__class__.__name__} ({direction})")
        # Capture TTFB from MetricsFrame for realtime/live models
        if isinstance(frame, MetricsFrame):
            for md in frame.data:
                if isinstance(md, TTFBMetricsData):
                    recorder = self._recorder_accessor()
                    if recorder:
                        recorder.record_ttfb(md.value)
        await self.push_frame(frame, direction)


class RealtimePipeline(BasePipeline):
    """Pipeline for OpenAI Realtime and Gemini Live models.

    This pipeline handles speech-to-speech models with:
    - Paced audio input at realtime pace
    - Server-side VAD for turn detection
    - Transcript-based end-of-turn detection
    - Reconnection handling for Gemini Live 10-minute timeout
    """

    requires_service = True

    def __init__(self, benchmark):
        super().__init__(benchmark)
        self.context_aggregator = None
        self.paced_input = None
        self.transcript = None
        self.assistant_shim = None
        self.current_turn_audio_path: Optional[str] = None
        self.needs_turn_retry: bool = False
        self.reconnection_grace_until: float = 0

    def _is_gemini_live(self) -> bool:
        """Check if current model is Gemini Live."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return (m.startswith("gemini") or m.startswith("models/gemini")) and (
            "live" in m or "native-audio" in m
        )

    def _is_openai_realtime(self) -> bool:
        """Check if current model is OpenAI Realtime."""
        if not self.model_name:
            return False
        m = self.model_name.lower()
        return "realtime" in m and m.startswith("gpt")

    def _get_audio_path_for_turn(self, turn_index: int) -> Optional[str]:
        """Get the audio file path for a turn.

        Prefers benchmark.get_audio_path() if available, falls back to
        the turn's audio_file field.

        Args:
            turn_index: The effective turn index (index into effective_turns).

        Returns:
            Path to audio file as string, or None if not available.
        """
        # Try benchmark's get_audio_path method first (uses audio_dir)
        if hasattr(self.benchmark, "get_audio_path"):
            actual_index = self._get_actual_turn_index(turn_index)
            path = self.benchmark.get_audio_path(actual_index)
            if path and path.exists():
                return str(path)

        # Fall back to turn's audio_file field
        turn = self.effective_turns[turn_index]
        return turn.get("audio_file")

    def _create_llm(
        self, service_class: Optional[type], model: str
    ) -> FrameProcessor:
        """Create LLM service with proper configuration for realtime models.

        For OpenAI Realtime, we must pass session_properties with turn_detection
        config at construction time. The server-side VAD settings prevent
        client-side interruptions from truncating responses.

        For Gemini Live, we use GeminiLiveLLMServiceWithReconnection and pass
        VAD parameters through the input params.
        """
        if service_class is None:
            raise ValueError("--service is required for this pipeline")

        class_name = service_class.__name__
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        if "OpenAIRealtime" in class_name:
            # OpenAI Realtime: Configure server-side VAD to prevent interruptions
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable is required")

            session_props = rt_events.SessionProperties(
                instructions=system_instruction,
                tools=tools,
                turn_detection={
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 1500,
                },
            )
            return service_class(
                api_key=api_key,
                model=model,
                system_instruction=system_instruction,
                session_properties=session_props,
            )
        else:
            # For Gemini Live and others, use base class implementation
            return super()._create_llm(service_class, model)

    def _setup_context(self) -> None:
        """Create LLMContext with system prompt and tools."""
        system_instruction = getattr(self.benchmark, "system_instruction", "")
        tools = getattr(self.benchmark, "tools_schema", None)

        # Both OpenAI Realtime and Gemini Live read the system instruction from
        # an LLMContextFrame. The pipecat service extracts the system message
        # and applies it via session properties (OpenAI) or context (Gemini).
        messages = [{"role": "system", "content": system_instruction}]

        self.context = LLMContext(messages, tools=tools)
        self.context_aggregator = LLMContextAggregatorPair(self.context)

    def _setup_llm(self) -> None:
        """Configure LLM and set up reconnection callbacks for Gemini Live."""
        self.llm.register_function(None, self._function_catchall)

        # Set up reconnection callbacks for Gemini Live
        if self._is_gemini_live() and isinstance(self.llm, GeminiLiveLLMServiceWithReconnection):
            self.llm._on_reconnecting = self._on_gemini_reconnecting
            self.llm._on_reconnected = self._on_gemini_reconnected

    def _on_gemini_reconnecting(self):
        """Called when Gemini Live starts reconnecting due to session timeout."""
        logger.info(f"Gemini reconnecting: pausing audio, turn {self.turn_idx} will be retried")
        self.needs_turn_retry = True
        # Pause audio input to avoid sending audio during reconnection
        self.paced_input.pause()
        # Clear the transcript buffer to discard partial responses from before reconnection
        self.assistant_shim.clear_buffer()
        # Set grace period to ignore stale TTSStoppedFrame events that arrive after reconnection
        self.reconnection_grace_until = time.monotonic() + 10.0
        logger.info(f"Set reconnection grace period until {self.reconnection_grace_until}")

    def _on_gemini_reconnected(self):
        """Called when Gemini Live reconnection completes."""
        logger.info(f"Gemini reconnected: scheduling turn {self.turn_idx} retry")
        # Resume audio input
        self.paced_input.signal_ready()
        # Schedule a task to re-queue the current turn's audio after a short delay
        asyncio.create_task(self._retry_current_turn_after_reconnection())

    async def _retry_current_turn_after_reconnection(self):
        """Re-queue the current turn's audio after reconnection."""
        if not self.needs_turn_retry:
            logger.info("No turn retry needed")
            return

        logger.info(f"Waiting 2s for connection to stabilize before retrying turn {self.turn_idx}")
        await asyncio.sleep(2.0)

        # Check if we still need to retry
        if not self.needs_turn_retry:
            logger.info("Turn retry cancelled (turn completed normally)")
            return

        # Get the audio path for the current turn
        audio_path = self.current_turn_audio_path or self._get_audio_path_for_turn(self.turn_idx)
        if audio_path:
            logger.info(f"Re-queuing audio for turn {self.turn_idx}: {audio_path}")
            try:
                self.paced_input.enqueue_wav_file(audio_path)
                self.needs_turn_retry = False
                logger.info(f"Successfully re-queued audio for turn {self.turn_idx}")
                # Wait for audio to finish then clear grace period
                await asyncio.sleep(5.0)
                self.reconnection_grace_until = 0
                logger.info("Cleared reconnection grace period - accepting new transcript updates")
            except Exception as e:
                logger.exception(f"Failed to re-queue audio for turn {self.turn_idx}: {e}")
        else:
            logger.warning(f"No audio path available for turn {self.turn_idx}, falling back to text")
            # Fall back to text
            await self.task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": turn["input"]}],
                        run_llm=False,
                    )
                ]
            )
            self.needs_turn_retry = False
            await asyncio.sleep(3.0)
            self.reconnection_grace_until = 0
            logger.info("Cleared reconnection grace period (text fallback)")

    def _build_task(self) -> None:
        """Build the pipeline with paced input and transcript processors."""

        def recorder_accessor():
            return self.recorder

        def duplicate_ids_accessor():
            return self._duplicate_tool_call_ids

        # Determine sample rate from first audio file
        default_sr = 24000
        t0_audio = self._get_audio_path_for_turn(0)
        if t0_audio:
            try:
                _, t0_sr = sf.read(t0_audio, dtype="int16", always_2d=True)
                default_sr = int(t0_sr)
            except Exception as e:
                logger.warning(f"Could not read sample rate from {t0_audio}: {e}")

        # Create paced input transport
        input_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=default_sr,
            audio_in_channels=1,
            audio_in_passthrough=True,
        )
        self.paced_input = PacedInputTransport(
            input_params,
            pre_roll_ms=100,
            continuous_silence=True,
        )

        # Create transcript processors
        self.transcript = TranscriptProcessor()
        self.assistant_shim = TTSStoppedAssistantTranscriptProcessor()

        # Register event handler for transcript updates
        @self.assistant_shim.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            # Check grace period
            if time.monotonic() < self.reconnection_grace_until:
                logger.warning(
                    f"Ignoring transcript update during reconnection grace period "
                    f"(until {self.reconnection_grace_until})"
                )
                return

            for msg in frame.messages:
                if isinstance(msg, TranscriptionMessage) and getattr(msg, "role", None) == "assistant":
                    timestamp = f"[{msg.timestamp}] " if msg.timestamp else ""
                    line = f"{timestamp}{msg.role}: {msg.content}"
                    logger.info(f"Transcript: {line}")
                    # Clear retry flag - turn completed successfully
                    self.needs_turn_retry = False
                    # Small delay to let downstream settle
                    await asyncio.sleep(1.0)
                    # Pass the assistant text directly
                    await self._on_turn_end(msg.content)

        llm_logger = LLMFrameLogger(recorder_accessor)

        pipeline = Pipeline(
            [
                self.paced_input,
                self.context_aggregator.user(),
                self.transcript.user(),
                self.llm,
                llm_logger,
                ToolCallRecorder(recorder_accessor, duplicate_ids_accessor),
                self.assistant_shim,
                self.context_aggregator.assistant(),
            ]
        )

        self.task = PipelineTask(
            pipeline,
            idle_timeout_secs=45,
            idle_timeout_frames=(MetricsFrame,),
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

    async def _queue_first_turn(self) -> None:
        """Queue audio for the first turn."""
        # For Gemini Live, push context frame to initialize the LLM with system
        # instruction and tools. This triggers ONE reconnect at startup.
        # For OpenAI Realtime, DO NOT send a context frame - it would force an
        # early response.create. OpenAI gets its config via session_properties
        # at construction time.
        if self._is_gemini_live():
            await self.task.queue_frames([LLMContextFrame(self.context)])

        # Give the pipeline a moment to start
        await asyncio.sleep(1.0)

        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        self.current_turn_audio_path = audio_path

        if audio_path:
            try:
                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"Queued paced audio for first turn: {audio_path}")
            except Exception as e:
                logger.exception(f"Failed to queue audio from {audio_path}: {e}")
                self.current_turn_audio_path = None
                # Fall back to text
                if self._is_gemini_live():
                    await self.task.queue_frames(
                        [
                            LLMMessagesAppendFrame(
                                messages=[{"role": "user", "content": turn["input"]}]
                            )
                        ]
                    )
                else:
                    await self.task.queue_frames([LLMRunFrame()])
        else:
            # No audio file, use text
            if self._is_gemini_live():
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}]
                        )
                    ]
                )
            else:
                await self.task.queue_frames([LLMRunFrame()])

    async def _queue_next_turn(self) -> None:
        """Queue audio or text for the next turn."""
        turn = self._get_current_turn()
        audio_path = self._get_audio_path_for_turn(self.turn_idx)
        self.current_turn_audio_path = audio_path

        if audio_path:
            try:
                self.paced_input.enqueue_wav_file(audio_path)
                logger.info(f"Queued paced audio for turn {self.turn_idx}: {audio_path}")
            except Exception as e:
                logger.exception(f"Failed to queue audio for turn {self.turn_idx}: {e}")
                audio_path = None

        if not audio_path:
            self.current_turn_audio_path = None
            # Fall back to text
            if self._is_gemini_live():
                await self.task.queue_frames(
                    [
                        LLMMessagesAppendFrame(
                            messages=[{"role": "user", "content": turn["input"]}],
                            run_llm=False,
                        )
                    ]
                )
            else:
                # OpenAI Realtime fallback
                self.context.add_messages([{"role": "user", "content": turn["input"]}])
                await self.task.queue_frames([LLMRunFrame()])
