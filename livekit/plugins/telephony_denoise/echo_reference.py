"""Paces outbound agent audio into the echo canceller as its reference."""

from __future__ import annotations

import asyncio
import contextlib
import time

from livekit import rtc
from livekit.agents.voice import io

from .log import logger
from .processor import TelephonyDenoiser

_LOOKAHEAD_SECONDS = 0.02

# Frames waiting to be paced into the canceller. A second of audio is far more
# than the pacer can fall behind in practice; past that the reference is stale
# enough to be useless, and holding it would only grow without limit.
_MAX_QUEUED_FRAMES = 100


class EchoReferenceTap(io.AudioOutput):
    """Passes agent audio through untouched while copying it to the canceller.

    Insert it in front of the existing room output, after `AgentSession.start`:

        session.output.audio = EchoReferenceTap(
            denoiser, next_in_chain=session.output.audio
        )
    """

    def __init__(
        self,
        denoiser: TelephonyDenoiser,
        *,
        next_in_chain: io.AudioOutput,
    ) -> None:
        super().__init__(
            label="EchoReferenceTap",
            next_in_chain=next_in_chain,
            capabilities=io.AudioOutputCapabilities(pause=True),
            # Inherit the requirement rather than reporting None, which would
            # tell the session any rate is fine and let TTS reach a sink that
            # cannot resample it.
            sample_rate=next_in_chain.sample_rate,
        )
        assert self.next_in_chain is not None
        self._denoiser = denoiser
        self._queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()
        self._pacer: asyncio.Task[None] | None = None
        self._playout_clock = 0.0
        self._paused = False

    @property
    def _sink(self) -> io.AudioOutput:
        # The base class may wrap a bare leaf in a proxy so it can be swapped
        # later, so read the chain rather than caching what was passed in.
        sink = self.next_in_chain
        assert sink is not None
        return sink

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

        if self._queue.qsize() >= _MAX_QUEUED_FRAMES:
            # Drop the stalest frame: a reference that late no longer lines up
            # with any echo still arriving.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(frame)

        # Only pace while playing. Starting here unconditionally used to undo
        # the `pause()` below on the very next frame.
        if not self._paused:
            self._start_pacer()

        await self._sink.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self._sink.flush()

    def clear_buffer(self) -> None:
        # The agent was interrupted, so queued audio will never reach the
        # caller. Feeding it as a reference would make the canceller hunt for an
        # echo that does not exist.
        self._drain()
        self._playout_clock = 0.0
        self._sink.clear_buffer()

    def pause(self) -> None:
        # Playout has stopped, so the pacer must stop too. Left running it would
        # keep feeding the canceller a reference for audio the caller is not
        # hearing yet, and the echo path would drift by the length of the pause.
        self._paused = True
        self._stop_pacer()
        super().pause()

    def resume(self) -> None:
        # Restart the virtual clock from the present; queued frames are still
        # pending and now line up with playout resuming.
        self._paused = False
        self._playout_clock = 0.0
        self._start_pacer()
        super().resume()

    def on_detached(self) -> None:
        self._stop_pacer()
        super().on_detached()

    async def aclose(self) -> None:
        """Stop pacing and wait for the task to unwind."""

        pacer, self._pacer = self._pacer, None
        self._drain()
        if pacer is not None:
            pacer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pacer

    def _start_pacer(self) -> None:
        if self._pacer is None or self._pacer.done():
            self._pacer = asyncio.create_task(self._pace())

    def _stop_pacer(self) -> None:
        if self._pacer is not None:
            self._pacer.cancel()
            self._pacer = None

    def _drain(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def _pace(self) -> None:
        try:
            while True:
                frame = await self._queue.get()

                now = time.monotonic()
                if self._playout_clock < now:
                    # First frame of an utterance, or we fell behind: restart
                    # the virtual clock from the present.
                    self._playout_clock = now

                self._denoiser.push_render_frame(frame)
                self._playout_clock += frame.duration

                # Stay slightly ahead of real time; the canceller wants the
                # reference before the echo arrives, never after.
                sleep_for = self._playout_clock - time.monotonic() - _LOOKAHEAD_SECONDS
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "echo reference pacer stopped, echo cancellation will degrade"
            )
