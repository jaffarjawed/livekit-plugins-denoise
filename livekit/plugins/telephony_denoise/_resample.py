"""Low-latency streaming resampler for the hop to and from the model's 48 kHz.

DeepFilterNet3 only runs at 48 kHz, so every call is resampled up and back down
again. That round trip, not the model, used to dominate end-to-end delay: soxr
picks its filter from a quality ladder tuned for offline work, where the only
settings with a real anti-alias filter cost 75-140 ms, and the cheap settings
have almost no stopband at all.

We do not need a general-purpose resampler. We need one good transition, from
the top of the voice band up to Nyquist, and that filter is short: a Kaiser
design hitting 80 dB of stopband lands around 10 ms of round-trip delay at
8 kHz and less above that, which is an order of magnitude better.

Delay is forced to a whole number of samples on both sides, so the pipeline
stays sample-aligned and a caller can reason about it exactly. That alignment
is what bounds the usable rate pairs: it costs a filter of `2 * L * M + 1` taps
in the worst case, so rates that are nearly but not exactly related (8000 to
8001, say) are rejected outright rather than quietly building a filter with
tens of millions of taps on the audio thread.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Keep the passband flat to this fraction of the lower Nyquist. At 8 kHz that
# puts the edge at 3520 Hz, above the 3400 Hz telephony band; at 16 kHz it
# clears wideband speech.
_PASSBAND = 0.88
_STOPBAND_DB = 80.0

# Taps per polyphase branch, which is the per-output-sample multiply count.
# Supported telephony rates land under 100; the cap only fires for rate pairs
# whose alignment blows the filter up, and those are rejected rather than run.
_MAX_PHASE_TAPS = 1024

# Elements per gather in the general path, to keep the temporary bounded no
# matter how long a block the caller hands over.
_CHUNK_ELEMS = 1 << 16


def _kaiser_beta(atten_db: float) -> float:
    if atten_db > 50:
        return 0.1102 * (atten_db - 8.7)
    if atten_db >= 21:
        return 0.5842 * (atten_db - 21) ** 0.4 + 0.07886 * (atten_db - 21)
    return 0.0


def _kaiser_size(
    internal_rate: int, pass_hz: float, stop_hz: float, atten_db: float, align: int
) -> tuple[int, int]:
    """Length and group delay of the design, without building it.

    Separate from `_design` so the cost of a rate pair can be checked before
    anything is allocated.
    """

    width = 2 * np.pi * (stop_hz - pass_hz) / internal_rate
    taps = int(np.ceil((atten_db - 8) / (2.285 * width))) + 1
    # Round the delay up to a multiple of `align` so it divides evenly into both
    # rates, and keep the filter symmetric (odd length).
    delay = -(-((taps - 1) // 2 + 1) // align) * align
    return 2 * delay + 1, delay


@lru_cache(maxsize=32)
def _design(
    internal_rate: int, pass_hz: float, stop_hz: float, atten_db: float, align: int
):
    """Kaiser-windowed sinc lowpass whose group delay is a multiple of `align`."""

    taps, delay = _kaiser_size(internal_rate, pass_hz, stop_hz, atten_db, align)

    cutoff = 0.5 * (pass_hz + stop_hz) / internal_rate
    k = np.arange(taps) - delay
    h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.kaiser(taps, _kaiser_beta(atten_db))
    return h / h.sum(), delay


class StreamResampler:
    """Rational resampler that keeps its filter state across blocks.

    Feed it whatever block sizes arrive; it returns however many output samples
    are ready. `delay` is the constant group delay, in samples at the internal
    rate, and divides evenly into both the input and the output rate.

    Input must be mono float audio. One instance carries the filter state for
    one stream and is *not* thread-safe; give each stream its own.
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        in_rate, out_rate = int(in_rate), int(out_rate)
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError(
                f"sample rates must be positive, got {in_rate} -> {out_rate}"
            )

        self.in_rate, self.out_rate = in_rate, out_rate
        divisor = gcd(in_rate, out_rate)
        self.L = out_rate // divisor
        self.M = in_rate // divisor

        # Nothing to do, and designing a real lowpass here would only shave the
        # top of the band off audio that is already at the right rate.
        self.identity = in_rate == out_rate
        if self.identity:
            self.P = 1
            self.delay = 0
            self._history = np.zeros(0, dtype=np.float32)
            return

        nyquist = min(in_rate, out_rate) / 2.0
        align = self.L * self.M
        internal_rate = in_rate * self.L
        taps_len, _ = _kaiser_size(
            internal_rate, _PASSBAND * nyquist, nyquist, _STOPBAND_DB, align
        )
        if -(-taps_len // self.L) > _MAX_PHASE_TAPS:
            raise ValueError(
                f"{in_rate} Hz -> {out_rate} Hz needs {-(-taps_len // self.L)} taps per phase, "
                f"over the {_MAX_PHASE_TAPS} budget; the rates are too close to unrelated. "
                "Pick rates with a larger common divisor."
            )

        taps, self.delay = _design(
            internal_rate, _PASSBAND * nyquist, nyquist, _STOPBAND_DB, align
        )
        taps = taps * self.L  # interpolation gain

        taps = np.concatenate((taps, np.zeros((-len(taps)) % self.L)))
        self.P = len(taps) // self.L
        phases = taps.reshape(self.P, self.L).T

        # The fast paths feed BLAS from (P, L); the general path gathers rows
        # from (L, P). Only one layout is ever used, so only one is kept.
        if self.M == 1 or self.L == 1:
            self._phases = np.ascontiguousarray(phases.T, dtype=np.float32)  # (P, L)
        else:
            self._phases = np.ascontiguousarray(phases, dtype=np.float32)  # (L, P)

        self._history = np.zeros(self.P - 1, dtype=np.float32)
        self._n_in = 0
        self._n_out = 0

    @property
    def delay_in(self) -> int:
        """Group delay counted in input samples."""

        return 0 if self.identity else self.delay // self.L

    @property
    def delay_out(self) -> int:
        """Group delay counted in output samples."""

        return 0 if self.identity else self.delay // self.M

    @property
    def delay_seconds(self) -> float:
        """Group delay in seconds. `delay` itself counts internal-rate samples."""

        return 0.0 if self.identity else self.delay / (self.in_rate * self.L)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Resample one block. Returns the samples that are ready, possibly none."""

        if x.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {x.shape}")
        if not np.issubdtype(x.dtype, np.floating):
            # Integer PCM would sail through `astype` scaled by 32768 and blow
            # out every downstream stage, so refuse it at the door.
            raise TypeError(f"expected float audio in [-1, 1], got {x.dtype}")

        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.identity:
            return x.copy()
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        buf = np.concatenate((self._history, x))
        base = self._n_in - (self.P - 1)
        last = self._n_in + x.size - 1

        # Output n reads input up to index (n*M)//L, so this is the first n we
        # cannot compute yet.
        end = -(-(last + 1) * self.L // self.M)
        if end <= self._n_out:
            self._advance(buf, x.size)
            return np.zeros(0, dtype=np.float32)

        n = np.arange(self._n_out, end, dtype=np.int64)
        j = n * self.M
        phase = j % self.L
        newest = j // self.L

        # Every output reads a P-sample window ending at `newest`; the oldest one
        # must still be in `buf`. If this trips, the counters and the history
        # have drifted apart and the audio below would be silently wrong.
        first = int(newest[0]) - base - self.P + 1
        assert first >= 0, f"resampler history underrun: first={first}"

        windows = sliding_window_view(buf, self.P)[:, ::-1]
        if self.M == 1:
            # Pure upsample: every phase fires for every input sample, so the
            # whole block is one matrix product.
            stop = int(newest[-1]) - base - self.P + 2
            out = (windows[first:stop] @ self._phases).ravel()
            out = out[int(phase[0]) : int(phase[0]) + n.size]
        elif self.L == 1:
            # Pure decimate: one phase, every M-th window.
            last_row = int(newest[-1]) - base - self.P + 1
            out = windows[first : last_row + 1 : self.M] @ self._phases[:, 0]
        else:
            rows = newest - base - self.P + 1
            out = np.empty(n.size, dtype=np.float32)
            # Gathering every window at once is the largest allocation in the
            # class, and it scales with the caller's block length, so cap it.
            step = max(1, _CHUNK_ELEMS // self.P)
            for start in range(0, n.size, step):
                stop = min(start + step, n.size)
                np.einsum(
                    "ij,ij->i",
                    windows[rows[start:stop]],
                    self._phases[phase[start:stop]],
                    out=out[start:stop],
                )

        self._n_out = end
        self._advance(buf, x.size)
        return np.ascontiguousarray(out, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Push silence through to release the tail still inside the filter.

        For offline use. A live stream should not call this: it appends real
        latency and the next block would land after a gap of silence.
        """

        if self.identity:
            return np.zeros(0, dtype=np.float32)
        return self.process(np.zeros(self.delay_in, dtype=np.float32))

    def _advance(self, buf: np.ndarray, consumed: int) -> None:
        self._n_in += consumed
        if self.P > 1:
            self._history = buf[buf.size - (self.P - 1) :].copy()

        # L outputs consume exactly M inputs, so rebasing by that pair keeps the
        # counters bounded on a long call without shifting the phase.
        if self._n_out >= self.L:
            whole = self._n_out // self.L
            self._n_out -= whole * self.L
            self._n_in -= whole * self.M
