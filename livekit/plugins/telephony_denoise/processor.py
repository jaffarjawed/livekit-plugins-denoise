"""LiveKit frame processor for runtime telephony noise and echo suppression."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from livekit import rtc

from ._buffers import Int16Buffer, to_mono
from .log import logger
from .neural import NeuralEnhancer

_BLOCK_MS = 10
"""The WebRTC APM contract: audio must be handed over in 10 ms blocks."""

_RATE_QUANTUM = 1000 // _BLOCK_MS
"""A supported rate must divide by this, so a block is a whole number of samples."""

_MAX_PENDING_RENDER_MS = 2000.0
"""How much far-end audio to hold before the first inbound frame sets the format."""

Enhancer = Literal["deepfilter", "webrtc"]


@dataclass
class DenoiseOptions:
    """Tuning for `TelephonyDenoiser`."""

    echo_cancellation: bool = True
    noise_suppression: bool = True
    high_pass_filter: bool = True
    auto_gain_control: bool = True

    enhancer: Enhancer = "deepfilter"
    """Noise suppressor for unknown caller environments.

    `deepfilter` (default) runs DeepFilterNet3 after AEC and handles gym / café /
    babble as well as line hiss. `webrtc` keeps the classical WebRTC NS for
    minimum latency. Never stack both.
    """

    stream_delay_ms: int = 120
    """Round trip delay between us sending audio and its echo arriving back.

    AEC3 refines this internally, but a sane starting point matters on SIP where
    the PSTN leg adds far more delay than a WebRTC client would.
    """

    def __post_init__(self) -> None:
        if self.enhancer not in get_args(Enhancer):
            raise ValueError(
                f"enhancer must be one of {get_args(Enhancer)}, got {self.enhancer!r}"
            )
        if self.stream_delay_ms < 0:
            raise ValueError(
                f"stream_delay_ms must not be negative, got {self.stream_delay_ms}"
            )


class TelephonyDenoiser(rtc.FrameProcessor[rtc.AudioFrame]):
    """Suppresses background noise and acoustic echo on an inbound audio track.

    The inbound (caller) audio flows through `process`, which LiveKit calls for
    every frame. For echo cancellation the filter also needs the far-end
    reference, meaning the audio the agent sends to the caller; feed that in via
    `push_render_frame`, or let `EchoReferenceTap` do it for you.

    One instance holds the echo path and noise estimate for a single
    conversation. Build a new one per call; never share across calls.
    """

    def __init__(self, opts: DenoiseOptions | None = None) -> None:
        self._opts = opts or DenoiseOptions()
        self._enabled = True
        self._closed = False

        # The format we gave up on, if any. Held rather than a bare flag so a
        # renegotiation to a format we do support starts filtering again.
        self._degraded: tuple[int, int] | None = None
        self._logged: set[str] = set()

        # `process` runs on the room's event loop while `push_render_frame` is
        # driven by the TTS output path, which may be a different thread.
        self._lock = threading.Lock()

        self._apm: rtc.AudioProcessingModule | None = None
        self._rate = 0
        self._channels = 0
        self._block = 0

        self._capture_in = Int16Buffer()
        self._capture_out = Int16Buffer()

        self._render_in = Int16Buffer()
        self._render_resampler: rtc.AudioResampler | None = None
        self._render_resampler_rate = 0
        self._pending_render: list[rtc.AudioFrame] = []
        self._pending_render_ms = 0.0

        self._neural: NeuralEnhancer | None = None
        self._use_neural = False
        self._webrtc_ns = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def enhancer(self) -> Enhancer | None:
        """Active noise suppressor, or None if it is off or not yet negotiated.

        The backend is only known once the first frame has revealed the stream
        format, because that is when a DeepFilterNet failure would fall back.
        """

        if not self._opts.noise_suppression or self._apm is None:
            return None
        return "deepfilter" if self._use_neural else "webrtc"

    def prepare(self, sample_rate: int, num_channels: int = 1) -> bool:
        """Build the filter ahead of the first frame.

        Optional: the format is picked up from the first inbound frame anyway.
        Doing it up front keeps the model load off the first frame's deadline and
        lets far-end audio be used as a reference from the very first word.

        Returns False if the format is not supported, in which case audio will
        pass through untouched.
        """

        with self._lock:
            if self._closed:
                return False
            return self._ensure_apm(sample_rate, num_channels)

    def process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        """Filter one inbound frame. Returns a frame of the same shape."""

        return self._process(frame)

    def close(self) -> None:
        """Release the filter. Further frames pass through untouched."""

        self._close()

    def push_render_frame(self, frame: rtc.AudioFrame) -> None:
        """Supply far-end audio (what the agent is saying) as the AEC reference.

        Call this for every frame the agent publishes, as close to playout as
        possible. Without it, noise suppression still works but echo
        cancellation has nothing to subtract.
        """

        if self._closed or not self._opts.echo_cancellation:
            return

        try:
            with self._lock:
                # Re-check under the lock; `_close` may have run since.
                if self._closed:
                    return
                if self._apm is None:
                    self._stash_render(frame)
                    return
                self._feed_render(frame)
        except Exception:
            self._log_once(
                "render", "failed to push render frame, echo cancellation may degrade"
            )

    def _process(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._enabled or self._closed:
            return frame
        if (frame.sample_rate, frame.num_channels) == self._degraded:
            return frame

        try:
            with self._lock:
                if self._closed:
                    return frame
                return self._process_capture(frame)
        except Exception:
            self._log_once(
                "capture", "audio processing failed, passing frames through untouched"
            )
            return frame

    def _close(self) -> None:
        with self._lock:
            self._closed = True
            self._apm = None
            if self._neural is not None:
                self._neural.close()
                self._neural = None
            self._capture_in.clear()
            self._capture_out.clear()
            self._render_in.clear()
            self._render_resampler = None
            self._pending_render.clear()
            self._pending_render_ms = 0.0

    def _log_once(self, key: str, message: str) -> None:
        """Report a recurring failure once.

        These fire from the per-frame path, so an unfiltered `logger.exception`
        would emit a traceback every 10 ms for the rest of the call.
        """

        if key in self._logged:
            return
        self._logged.add(key)
        logger.exception("%s (further occurrences suppressed)", message)

    def _process_capture(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if not self._ensure_apm(frame.sample_rate, frame.num_channels):
            return frame

        wanted = frame.samples_per_channel * frame.num_channels
        self._capture_in.append(self._samples(frame))

        stride = self._block * self._channels
        try:
            while self._capture_in.size >= stride:
                self._capture_out.append(
                    self._process_block(self._capture_in.take(stride))
                )
        except Exception:
            # Whatever is queued is now out of step with the filter state. Left
            # in place it is never drained and shows up as latency that grows
            # for the rest of the call, so start the stream over instead.
            self._reset_stream_buffers()
            raise

        processed = self._capture_out.take(wanted)
        if processed.size < wanted:
            processed = np.concatenate(
                (processed, np.zeros(wanted - processed.size, dtype=np.int16))
            )
        return rtc.AudioFrame(
            processed.tobytes(),
            frame.sample_rate,
            frame.num_channels,
            frame.samples_per_channel,
        )

    @staticmethod
    def _samples(frame: rtc.AudioFrame) -> np.ndarray:
        """Declared samples of a frame, ignoring any slack in its backing buffer."""

        pcm = np.frombuffer(frame.data, dtype=np.int16)
        wanted = frame.samples_per_channel * frame.num_channels
        if pcm.size < wanted:
            raise ValueError(f"frame declares {wanted} samples but carries {pcm.size}")
        # A buffer longer than the declared frame would otherwise queue up as
        # latency that never drains.
        return pcm[:wanted]

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        assert self._apm is not None

        frame = rtc.AudioFrame(block.tobytes(), self._rate, self._channels, self._block)
        self._apm.process_stream(frame)
        out = np.frombuffer(frame.data, dtype=np.int16).copy()

        if self._use_neural and self._neural is not None:
            out = self._neural.process(out, self._rate)

        return out

    def _resolve_ns_backend(self) -> None:
        """Pick neural or WebRTC NS. Never enable both."""

        self._use_neural = False
        self._webrtc_ns = False
        if self._neural is not None:
            self._neural.close()
        self._neural = None

        if not self._opts.noise_suppression:
            return

        if self._opts.enhancer == "deepfilter":
            try:
                self._neural = NeuralEnhancer()
                self._use_neural = True
                return
            except Exception:
                # neural._get_model logs the first failure with a traceback.
                logger.warning(
                    "DeepFilterNet3 unavailable, using WebRTC noise suppression"
                )
                self._webrtc_ns = True
                return

        self._webrtc_ns = True

    def _ensure_apm(self, sample_rate: int, num_channels: int) -> bool:
        if (
            self._apm is not None
            and sample_rate == self._rate
            and num_channels == self._channels
        ):
            return True

        if not self._supported(sample_rate, num_channels):
            return False

        self._rate = sample_rate
        self._channels = num_channels
        self._block = sample_rate // _RATE_QUANTUM
        self._degraded = None

        self._resolve_ns_backend()

        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=self._opts.echo_cancellation,
            noise_suppression=self._webrtc_ns,
            high_pass_filter=self._opts.high_pass_filter,
            auto_gain_control=self._opts.auto_gain_control,
        )
        if self._opts.echo_cancellation:
            # Stored state, not a per-frame argument, and each call is a full
            # FFI round trip -- so set it here rather than every 10 ms.
            self._apm.set_stream_delay_ms(self._opts.stream_delay_ms)

        self._reset_stream_buffers()
        self._render_resampler = None
        self._render_resampler_rate = 0

        logger.info(
            "audio filter ready: %d Hz, %d ch, aec=%s ns=%s enhancer=%s agc=%s hpf=%s",
            sample_rate,
            num_channels,
            self._opts.echo_cancellation,
            self._opts.noise_suppression,
            self.enhancer or "off",
            self._opts.auto_gain_control,
            self._opts.high_pass_filter,
        )

        self._replay_render()
        return True

    def _supported(self, sample_rate: int, num_channels: int) -> bool:
        """Check the stream format, and step aside for the rest of the call if not.

        Degrading beats raising: an unsupported format is a property of the call,
        so raising would log a traceback for every frame until it ends.
        """

        reason = None
        if num_channels != 1:
            reason = f"{num_channels} channels; telephony denoising is mono only"
        elif sample_rate % _RATE_QUANTUM:
            reason = (
                f"{sample_rate} Hz, which is not a whole number of samples per "
                f"{_BLOCK_MS} ms block"
            )

        if reason is None:
            return True

        if self._degraded != (sample_rate, num_channels):
            self._degraded = (sample_rate, num_channels)
            logger.error("cannot filter %s; passing audio through unfiltered", reason)
        return False

    def _reset_stream_buffers(self) -> None:
        self._capture_in.clear()
        self._capture_out.clear()
        self._render_in.clear()

        # Prime the output with one block of silence so that every inbound frame
        # can be answered with an equal number of samples. Costs 10 ms of
        # added latency and keeps the stream sample-accurate. Neural cold-start
        # pads inside NeuralEnhancer so we do not need a larger prime here.
        self._capture_out.append(np.zeros(self._block * self._channels, dtype=np.int16))

    def _stash_render(self, frame: rtc.AudioFrame) -> None:
        """Hold far-end audio that arrived before the format was known.

        The agent usually speaks first, so without this the whole greeting is
        missing from the echo reference and its echo cannot be subtracted.
        """

        self._pending_render.append(frame)
        self._pending_render_ms += frame.duration * 1000.0
        while self._pending_render_ms > _MAX_PENDING_RENDER_MS and self._pending_render:
            self._pending_render_ms -= self._pending_render.pop(0).duration * 1000.0

    def _replay_render(self) -> None:
        pending, self._pending_render = self._pending_render, []
        self._pending_render_ms = 0.0
        if not pending or not self._opts.echo_cancellation:
            return
        try:
            for frame in pending:
                self._feed_render(frame)
        except Exception:
            self._log_once(
                "render", "failed to replay far-end audio into the echo canceller"
            )

    def _feed_render(self, frame: rtc.AudioFrame) -> None:
        assert self._apm is not None

        pcm = to_mono(self._samples(frame), frame.num_channels)

        if frame.sample_rate != self._rate:
            pcm = self._resample_render(pcm, frame.sample_rate)

        self._render_in.append(pcm)

        stride = self._block * self._channels
        while self._render_in.size >= stride:
            chunk = self._render_in.take(stride)
            self._apm.process_reverse_stream(
                rtc.AudioFrame(chunk.tobytes(), self._rate, self._channels, self._block)
            )

    def _resample_render(self, pcm: np.ndarray, src_rate: int) -> np.ndarray:
        chunks: list[np.ndarray] = []

        if (
            self._render_resampler is not None
            and self._render_resampler_rate != src_rate
        ):
            # Push out the tail still inside the old filter before dropping it,
            # otherwise the reference loses samples and the echo path shifts.
            chunks += [
                np.frombuffer(f.data, dtype=np.int16)
                for f in self._render_resampler.flush()
            ]
            self._render_resampler = None

        if self._render_resampler is None:
            self._render_resampler = rtc.AudioResampler(
                src_rate, self._rate, num_channels=self._channels
            )
            self._render_resampler_rate = src_rate

        chunks += [
            np.frombuffer(f.data, dtype=np.int16)
            for f in self._render_resampler.push(bytearray(pcm.tobytes()))
        ]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)
