import asyncio
from typing import Callable

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    MetricsFrame,
    LLMContextAssistantTimestampFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class NextTurn(FrameProcessor):
    """Processor that detects end-of-turn and triggers callbacks.

    Used in text pipelines to detect when the assistant has finished responding
    (via LLMContextAssistantTimestampFrame) and trigger the next turn.
    """

    def __init__(
        self, end_of_turn_callback: Callable, metrics_callback: Callable[[MetricsFrame], None]
    ):
        super().__init__()
        self.end_of_turn_callback = end_of_turn_callback
        self.metrics_callback = metrics_callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            self.metrics_callback(frame)

        # Treat assistant timestamp frame as end-of-turn marker
        if isinstance(frame, LLMContextAssistantTimestampFrame):
            logger.info("EOT (timestamp)")
            await self.end_of_turn_callback()
