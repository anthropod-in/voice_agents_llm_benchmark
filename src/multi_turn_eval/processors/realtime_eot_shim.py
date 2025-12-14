import asyncio

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMContextAssistantTimestampFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.time import time_now_iso8601


class RealtimeEOTShim(FrameProcessor):
    """Shim for OpenAI Realtime: inject assistant message + timestamp when
    the service omits LLMFullResponseStartFrame (so the assistant aggregator
    won't aggregate tokens or emit a timestamp).

    Sits between the LLM and the assistant context aggregator.
    """

    def __init__(self):
        super().__init__()

    async def push_eot_frames(self):
        await asyncio.sleep(0.5)
        ts = LLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
        await self.push_frame(ts)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            asyncio.create_task(self.push_eot_frames())

        await self.push_frame(frame, direction)
