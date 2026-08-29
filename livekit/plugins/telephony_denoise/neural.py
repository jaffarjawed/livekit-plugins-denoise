"""DeepFilterNet3 noise suppression for unknown telephony environments."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from ._resample import StreamResampler
from .log import logger

_PRIME_LIMIT = 64  # pre-roll blocks before we give up and warn

# How long to sit out after a failed load. Retrying per call would stall every
# call on the same slow failure; never retrying would let one flaky download at
# worker start disable neural suppression for the life of the process.
_RETRY_AFTER_SECONDS = 60.0

_model_lock = threading.Lock()
_shared_model: Any | None = None
_failed_at = 0.0


def _get_model() -> Any:
    """Load one ONNX session per process. Thread-safe to share; streams are not."""

    global _shared_model, _failed_at

    # Loading downloads the weights on first use, so hold the lock across the
    # whole thing: concurrent calls should wait, not each start a download.
    with _model_lock:
        if _shared_model is not None:
            return _shared_model

        waited = time.monotonic() - _failed_at
        if _failed_at and waited < _RETRY_AFTER_SECONDS:
            raise RuntimeError(
                f"DeepFilterNet3 unavailable, retrying in "
                f"{_RETRY_AFTER_SECONDS - waited:.0f}s"
            )

        try:
            from deepfilter_stream import DeepFilterModel

            # One ONNX thread per session so concurrent SIP calls do not thrash.
            _shared_model = DeepFilterModel(intra_op_num_threads=1)
        except Exception:
            _failed_at = time.monotonic()
            logger.exception("DeepFilterNet3 failed to load; falling back to WebRTC NS")
            raise

        _failed_at = 0.0
        logger.info(
            "DeepFilterNet3 model loaded: %d Hz, %d-sample frames",
            _shared_model.sample_rate,
            _shared_model.frame_size,
        )
        return _shared_model


def prewarm() -> None:
    """Download and load the model up front.

    The first call otherwise pays for a network fetch and an ONNX session build
    inside its first audio frame, on the event loop. Call this from a worker's
    setup hook.
    """

    _get_model()


class NeuralEnhancer:
    """Per-call DeepFilterNet stream. Always processes at 48 kHz internally.

    Feeding 8/16 kHz chunks through `process(sr=...)` builds a large resample
    buffer and adds hundreds of milliseconds of delay, so we resample ourselves
    and call `process_frame` on whole model frames.

    Delay across this stage is the model's own 32 ms lookahead plus a few ms of
    resampler group delay: measured with the model running, roughly 43 ms at
    48 kHz, 48 ms at 16 kHz and 53 ms at 8 kHz. The model dominates, which is
    the point of `_resample`; a general-purpose resampler put 8 kHz at 172 ms.
    The delay is constant for the life of the stream -- see
    tests/test_neural_plumbing.py, which pins the resampler part with the model
    stubbed out.

    One stream per call. Not thread-safe, and closing it is final.
    """

    def __init__(self) -> None:
        model = _get_model()
        try:
            self._stream = model.new_stream()
        except Exception:
            logger.exception("DeepFilterNet3 stream failed; falling back to WebRTC NS")
            raise

        # Read the geometry off the loaded graph. Hardcoding 512 meant a model
        # asset with a different hop raised on every single frame.
        self._model_rate = int(model.sample_rate)
        self._frame = int(model.frame_size)

        self._closed = False
        self._rate = 0
        self._up_rs: StreamResampler | None = None
        self._down_rs: StreamResampler | None = None
        self._up = np.zeros(0, dtype=np.float32)
        self._down = np.zeros(0, dtype=np.float32)

    def _configure(self, sample_rate: int) -> None:
        """Rebuild the resamplers and pre-roll the output buffer for a new rate."""

        self._rate = sample_rate
        self._stream.reset()
        self._up = np.zeros(0, dtype=np.float32)
        self._down = np.zeros(0, dtype=np.float32)

        # These carry their filter state across blocks. Resampling each block
        # independently restarts that state every time, and the transient at
        # each block edge is loud enough to hear as distortion. At the model's
        # own rate they are pass-throughs and cost nothing.
        self._up_rs = StreamResampler(sample_rate, self._model_rate)
        self._down_rs = StreamResampler(self._model_rate, sample_rate)

        # Output arrives in whole model frames and trails input by the
        # group delay of both resamplers, so the buffer starts empty and stays
        # that way for a while. Left alone it runs dry mid-call and we splice in
        # silence, which both clicks and pushes the audio permanently later --
        # that is how latency crept past 300 ms at 8 kHz. Push silence through
        # the whole chain now so the delay is flushed before real audio arrives,
        # and keep one frame in hand to absorb the frame-boundary jitter.
        quantum = self.quantum
        for _ in range(_PRIME_LIMIT):
            if self._down.size >= quantum:
                break
            self._pump(np.zeros(quantum, dtype=np.float32))
        else:
            logger.warning(
                "DeepFilterNet pre-roll did not fill at %d Hz; expect a brief warm-up",
                sample_rate,
            )

    @property
    def quantum(self) -> int:
        """One model frame expressed in caller-rate samples.

        The output buffer can only ever fall this far behind the input, whatever
        block size the caller uses, so it is also the pre-roll target.
        """

        return int(np.ceil(self._frame * self._rate / self._model_rate))

    def _pump(self, mono: np.ndarray) -> None:
        """Run one block of float32 audio through resample -> model -> resample."""

        assert self._up_rs is not None and self._down_rs is not None

        up = self._up_rs.process(mono)
        if up.size:
            self._up = np.concatenate((self._up, up))

        enhanced: list[np.ndarray] = []
        while self._up.size >= self._frame:
            frame = self._up[: self._frame]
            self._up = self._up[self._frame :]
            enhanced.append(self._stream.process_frame(frame))

        if not enhanced:
            return

        chunk = np.concatenate(enhanced).astype(np.float32)
        if not np.isfinite(chunk).all():
            # The resampler keeps a window of history, so a single NaN out of the
            # model would smear across every later block instead of one frame.
            logger.warning("DeepFilterNet3 returned non-finite samples; zeroing them")
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)

        down = self._down_rs.process(chunk)
        if down.size:
            self._down = np.concatenate((self._down, down))

    def process(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance a mono int16 block. Returns the same number of int16 samples."""

        if self._closed:
            raise RuntimeError("NeuralEnhancer is closed")
        if sample_rate != self._rate:
            self._configure(sample_rate)

        wanted = int(pcm.size)
        self._pump(pcm.astype(np.float32) * (1.0 / 32768.0))

        if self._down.size >= wanted:
            out = self._down[:wanted]
            self._down = self._down[wanted:]
        else:
            # The pre-roll is sized so this cannot happen in steady state. Reaching
            # it means the chain lost samples, so splice silence to keep the frame
            # contract rather than hand back a short block.
            pad = wanted - int(self._down.size)
            logger.warning("DeepFilterNet output ran dry, padding %d samples", pad)
            out = np.concatenate((self._down, np.zeros(pad, dtype=np.float32)))
            self._down = np.zeros(0, dtype=np.float32)

        return np.clip(np.rint(out * 32768.0), -32768, 32767).astype(np.int16)

    def close(self) -> None:
        """Release the stream. The enhancer cannot be used again afterwards."""

        if self._closed:
            return
        self._closed = True
        self._rate = 0
        self._up_rs = self._down_rs = None
        self._up = np.zeros(0, dtype=np.float32)
        self._down = np.zeros(0, dtype=np.float32)
        try:
            self._stream.reset()
        except Exception:
            logger.debug("DeepFilterNet stream reset failed on close", exc_info=True)
