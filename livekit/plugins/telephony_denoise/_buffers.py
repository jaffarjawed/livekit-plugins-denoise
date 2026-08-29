"""Small int16 PCM helpers used by the processing pipeline."""

from __future__ import annotations

import numpy as np


class Int16Buffer:
    """A FIFO of interleaved int16 samples."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)

    @property
    def size(self) -> int:
        return int(self._buf.size)

    def append(self, data: np.ndarray) -> None:
        # `np.concatenate(..., dtype=np.int16)` casts same_kind, so an int32
        # caller would be truncated into garbage PCM without a word. Only int16
        # is ever correct here, so insist on it.
        if data.dtype != np.int16:
            raise TypeError(f"expected int16 samples, got {data.dtype}")
        self._buf = np.concatenate((self._buf, data))

    def take(self, count: int) -> np.ndarray:
        out = self._buf[:count].copy()
        self._buf = self._buf[count:]
        return out

    def clear(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)


def to_mono(pcm: np.ndarray, src_channels: int) -> np.ndarray:
    """Downmix interleaved int16 audio when needed."""

    if src_channels < 1:
        raise ValueError(f"src_channels must be positive, got {src_channels}")
    if src_channels == 1:
        return pcm
    if pcm.size % src_channels:
        raise ValueError(
            f"{pcm.size} samples is not a whole number of {src_channels}ch frames"
        )

    frames = pcm.reshape(-1, src_channels).astype(np.int32)
    # Round rather than truncate: `astype` rounds toward zero, which biases
    # every downmixed sample toward silence and adds a DC-free crackle.
    mean = frames.sum(axis=1) / src_channels
    return np.rint(mean).astype(np.int16)
