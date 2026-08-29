from __future__ import annotations

import pytest
from livekit import rtc
from livekit.plugins import telephony_denoise


def _frame(*, rate: int = 16_000, channels: int = 1, samples: int = 160) -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\x00\x00" * samples * channels, rate, channels, samples)


def test_options_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="enhancer"):
        telephony_denoise.DenoiseOptions(enhancer="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be negative"):
        telephony_denoise.DenoiseOptions(stream_delay_ms=-1)


def test_webrtc_processor_preserves_frame_contract() -> None:
    denoiser = telephony_denoise.TelephonyDenoiser(
        telephony_denoise.DenoiseOptions(enhancer="webrtc")
    )
    try:
        assert denoiser.prepare(16_000)
        frame = _frame()
        result = denoiser.process(frame)
        assert (result.sample_rate, result.num_channels, result.samples_per_channel) == (
            frame.sample_rate,
            frame.num_channels,
            frame.samples_per_channel,
        )
        assert len(result.data) == len(frame.data)
        assert denoiser.enhancer == "webrtc"
    finally:
        denoiser.close()


def test_unsupported_stereo_frame_passes_through() -> None:
    denoiser = telephony_denoise.TelephonyDenoiser(
        telephony_denoise.DenoiseOptions(enhancer="webrtc")
    )
    try:
        frame = _frame(channels=2)
        assert denoiser.process(frame) is frame
    finally:
        denoiser.close()


def test_disabled_noise_suppression_has_no_enhancer() -> None:
    denoiser = telephony_denoise.TelephonyDenoiser(
        telephony_denoise.DenoiseOptions(noise_suppression=False)
    )
    try:
        assert denoiser.prepare(16_000)
        assert denoiser.enhancer is None
    finally:
        denoiser.close()
