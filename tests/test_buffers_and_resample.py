from __future__ import annotations

import numpy as np
import pytest

from livekit.plugins.telephony_denoise._buffers import Int16Buffer, to_mono
from livekit.plugins.telephony_denoise._resample import StreamResampler


def test_int16_buffer_is_fifo_and_rejects_wrong_dtype() -> None:
    buffer = Int16Buffer()
    buffer.append(np.array([1, 2, 3], dtype=np.int16))
    assert buffer.take(2).tolist() == [1, 2]
    assert buffer.take(2).tolist() == [3]
    with pytest.raises(TypeError, match="int16"):
        buffer.append(np.array([1], dtype=np.int32))


def test_to_mono_rounds_interleaved_samples() -> None:
    pcm = np.array([1, 2, -3, -4], dtype=np.int16)
    assert to_mono(pcm, 2).tolist() == [2, -4]


def test_identity_resampler_is_a_copy() -> None:
    source = np.array([0.25, -0.5], dtype=np.float32)
    result = StreamResampler(16_000, 16_000).process(source)
    assert np.array_equal(result, source)
    assert result is not source


def test_resampler_rejects_non_float_audio() -> None:
    resampler = StreamResampler(8_000, 48_000)
    with pytest.raises(TypeError, match="float"):
        resampler.process(np.array([1], dtype=np.int16))
