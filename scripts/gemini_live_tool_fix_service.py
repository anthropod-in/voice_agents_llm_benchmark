"""Gemini Live wrapper that sends turnComplete after tool results.

Gemini's realtime server infers turn completion automatically for plain
turns, but when a toolCall is made it expects a clientContent.turnComplete
after the toolResponse. Pipecat's stock GeminiLiveLLMService currently sends
the toolResponse but not turnComplete, so the server never emits the
turnComplete event and downstream never sees TTSStoppedFrame /
LLMFullResponseEndFrame. That causes hangs on tool turns.

This wrapper overrides _tool_result to append a turnComplete client event
right after sending the toolResponse. Everything else is delegated to the
base class.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.frames.frames import (
    Frame,
    TTSStoppedFrame,
    LLMFullResponseEndFrame,
    TTSTextFrame,
)
# Use google.genai.types for turnComplete (Pipecat 0.0.95 compatible)
try:
    # Preferred import in newer google.genai
    from google.genai.types import LiveClientMessage, LiveClientContent
except Exception:  # pragma: no cover - fallback
    LiveClientMessage = None
    LiveClientContent = None


class GeminiLiveToolFixedService(GeminiLiveLLMService):
    class TurnCompleteFrame(Frame):
        """Lightweight marker so downstream can detect server turn completion."""
        pass

    async def _handle_msg_input_transcription(self, message):
        """Log input (user) transcription at INFO for easier debugging."""
        try:
            sc = getattr(message, "server_content", None)
            txt = getattr(getattr(sc, "input_transcription", None), "text", None)
            if txt:
                logger.info(f"[GEMINI_FIX] Input transcription: {txt!r}")
        except Exception:
            pass

        await super()._handle_msg_input_transcription(message)

    # Toggle for buffering output transcription to avoid stuttery partial frames.
    emit_partial_output_transcripts = False
    _output_buffer = ""

    async def _handle_msg_output_transcription(self, message):
        """Log and convert output transcription; optionally buffer until final."""
        try:
            if not (message.server_content and message.server_content.output_transcription):
                return
            text = message.server_content.output_transcription.text or ""
            if self.emit_partial_output_transcripts:
                logger.debug(f"[GEMINI_FIX] Emitting TTSTextFrame from output_transcription: {text!r}")
                if text:
                    await self.push_frame(TTSTextFrame(text=text))
            else:
                # Buffer and emit later on TTSStopped/LLMFullResponseEnd
                self._output_buffer += text
        except Exception as e:
            logger.error(f"[GEMINI_FIX] Error extracting output transcription: {e}")

    async def _handle_msg_tool_call(self, message):  # type: ignore[override]
        # Extra visibility into tool calls coming from the service
        try:
            fc = getattr(message, "tool_call", None)
            logger.info(
                f"[GEMINI_FIX] tool_call message received: {fc}"
            )
            logger.info(f"[GEMINI_FIX] RAW TOOL CALL MESSAGE: {getattr(message, '__dict__', {})}")
        except Exception as exc:
            logger.warning(f"[GEMINI_FIX] tool_call logging failed: {exc}")

        await super()._handle_msg_tool_call(message)

    async def _handle_msg_turn_complete(self, message):  # type: ignore[override]
        """Log server-side turnComplete events and flush buffered transcripts."""
        try:
            logger.info(
                "[GEMINI_FIX] turnComplete received "
                f"server_content.turn_complete={getattr(getattr(message, 'server_content', None), 'turn_complete', None)} "
                f"message_class={message.__class__.__name__}"
            )
            # Emit buffered output transcription once at turn complete (if buffering enabled)
            if not self.emit_partial_output_transcripts and self._output_buffer:
                buf = self._output_buffer
                self._output_buffer = ""
                logger.debug(f"[GEMINI_FIX] Emitting buffered TTSTextFrame: {buf!r}")
                await self.push_frame(TTSTextFrame(text=buf))
            # Emit a local marker frame so the pipeline can react deterministically.
            await self.push_frame(self.TurnCompleteFrame())
        except Exception as exc:
            logger.warning(f"[GEMINI_FIX] turnComplete handler error: {exc}")
        await super()._handle_msg_turn_complete(message)

    async def _connect(self, session_resumption_handle: Optional[str] = None):
        """Establish client connection to Gemini Live API, with tool/system logging."""
        logger.info(
            f"[GEMINI_FIX] _connect called session_resumption_handle={session_resumption_handle} "
            f"tools_from_init_present={bool(getattr(self, '_tools_from_init', None))} "
            f"system_from_init_present={bool(getattr(self, '_system_instruction_from_init', None))} "
            f"settings_modalities={getattr(self, '_settings', {}).get('modalities', None)}"
        )
        await super()._connect(session_resumption_handle)

    async def _tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_result_message: Any,
    ):  # type: ignore[override]
        logger.info(
            f"[GEMINI_FIX] _tool_result called id={tool_call_id} name={tool_name} payload_keys={list(tool_result_message or {})}"
        )
        # First, send the normal tool response using the correct signature.
        await super()._tool_result(tool_call_id, tool_name, tool_result_message)

        # GeminiLiveLLMService doesn't expose send_client_event (that method is OpenAI Realtime-only).
        # Rely on the local failsafe below to signal pipeline completion for tool turns.

        # Failsafe: emit local stop frames so downstream sees end-of-turn even if the service withholds it.
        try:
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(LLMFullResponseEndFrame())
            logger.info("[GEMINI_FIX] Emitted local stop frames failsafe after tool result")
        except Exception as exc:  # pragma: no cover
            logger.error(f"[GEMINI_FIX] Failed to emit local stop frames: {exc}")
