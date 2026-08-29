# Technical notes

## Choosing an enhancer

`enhancer="deepfilter"` is the default and generally removes more complex
background noise. It downloads DeepFilterNet3 model weights (about 13 MB) on
first use, so call `telephony_denoise.prewarm()` during worker startup.

Use `enhancer="webrtc"` when lower latency and call density matter more than
neural suppression quality.

## Echo cancellation

Noise suppression works without additional wiring. Echo cancellation does not:
attach `EchoReferenceTap` after `session.start()` so the denoiser receives the
audio the caller actually hears. Create one denoiser per call; its echo and
noise state must not be shared between calls.

`stream_delay_ms=120` is a useful SIP starting point. Tune it using recordings
from your carrier route if returned agent audio remains audible.

## Audio and pipeline constraints

- Inbound telephony audio must be mono.
- Supported input rates are multiples of 100 Hz, such as 8, 16, 24, 32, and
  48 kHz. Unsupported audio passes through unfiltered rather than failing a
  call.
- Do not stack this denoiser with another echo or noise canceller; competing
  filters can degrade speech quality.
- If this plugin owns AGC, set LiveKit's `AudioInputOptions.auto_gain_control`
  to `False`.

## Dependencies and licensing

This package depends on `livekit`, `livekit-agents`, `numpy`, and
`deepfilter-stream`. It is MIT licensed. `deepfilter-stream` and DeepFilterNet
are dual-licensed under MIT or Apache-2.0; the required attribution is in the
project [NOTICE](../NOTICE). The DeepFilterNet3 model is downloaded by the
dependency and is not redistributed in this package.
