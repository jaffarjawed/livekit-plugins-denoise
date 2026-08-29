# livekit-plugins-denoise

Self-hosted noise suppression and acoustic echo cancellation for LiveKit SIP
agents.

- Removes background noise with DeepFilterNet3 or low-latency WebRTC noise
  suppression.
- Removes the agent's voice when it returns through a caller's handset or
  carrier path.
- Runs in your agent process: no per-minute denoising fee and no language
  dependency.

This is an independent, community-maintained package and is not affiliated
with or endorsed by LiveKit, Inc.

## Why this exists

LiveKit's managed `BVCTelephony` noise-cancellation option is billed per minute.
This package is a self-hosted, MIT-licensed alternative for teams that want
noise suppression and echo cancellation in their own agent process without a
per-minute denoising charge. You still pay for your own compute, telephony, and
other services.

## Install

```bash
python -m pip install livekit-plugins-denoise
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

## Enable noise suppression and echo cancellation

Create one `TelephonyDenoiser` for each call. `EchoReferenceTap` is required
for echo cancellation because it supplies the agent's outgoing audio as the
far-end reference.

```python
from livekit.agents import room_io
from livekit.plugins import telephony_denoise

denoiser = telephony_denoise.TelephonyDenoiser(
    telephony_denoise.DenoiseOptions(
        echo_cancellation=True,
        noise_suppression=True,
        high_pass_filter=True,
        auto_gain_control=True,
        enhancer="deepfilter",  # use "webrtc" for lower latency
        stream_delay_ms=120,
    )
)

await session.start(
    agent=agent,
    room=room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=denoiser,
            auto_gain_control=False,  # do not stack AGC
        ),
    ),
)

# Add this after session.start(); RoomIO replaces the output during startup.
session.output.audio = telephony_denoise.EchoReferenceTap(
    denoiser, next_in_chain=session.output.audio
)
```

To load the neural model before the first call:

```python
def setup(proc):
    telephony_denoise.prewarm()
```

See [examples/sip_agent.py](examples/sip_agent.py) for a complete SIP agent.

## Configuration

`DenoiseOptions` lets you control these features independently:

| Option | Default | Purpose |
| --- | --- | --- |
| `echo_cancellation` | `True` | Cancels the agent's returned audio; requires `EchoReferenceTap`. |
| `noise_suppression` | `True` | Removes background noise. |
| `enhancer` | `"deepfilter"` | Use `"webrtc"` when minimizing latency is more important. |
| `high_pass_filter` | `True` | Reduces low-frequency rumble and hum. |
| `auto_gain_control` | `True` | Helps keep quiet callers audible. |
| `stream_delay_ms` | `120` | Starting delay estimate for SIP echo paths. |

For tuning, supported formats, model download behavior, and operating notes, see
[Technical notes](docs/technical-notes.md).

## Community

Issues, documentation improvements, and pull requests are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md). This project is MIT licensed; dependency
attribution is in [NOTICE](NOTICE).

Developed and maintained by [Botcadence](https://botcadence.com).
