# livekit-plugins-denoise

Noise and echo suppression for LiveKit SIP calls that runs inside your own agent
process. It uses WebRTC audio processing plus DeepFilterNet3, so it has no
per-minute denoising fee and works independently of the caller's language.

This is an independent, community-maintained package. It is not affiliated with
or endorsed by LiveKit, Inc.

It plugs in exactly where the built-in filter would:

```python
from livekit.plugins import telephony_denoise

noise_cancellation=telephony_denoise.TelephonyDenoiser()
```

## Install

Install the published package:

```bash
python -m pip install livekit-plugins-denoise
```

For development from the monorepo root:

```bash
pip install -e backend/telephony-denoise
```

Or from this directory:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

## Use it

```python
from livekit.agents import Agent, AgentServer, AgentSession, room_io
from livekit.plugins import telephony_denoise

denoiser = telephony_denoise.TelephonyDenoiser()

await session.start(
    agent=Agent(instructions="..."),
    room=ctx.room,
    room_options=room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=denoiser,
            auto_gain_control=False,      # the denoiser already does this
        ),
    ),
)

# After start(), or RoomIO overwrites it.
session.output.audio = telephony_denoise.EchoReferenceTap(
    denoiser, next_in_chain=session.output.audio
)
```

Load the model once per worker rather than inside the first call's first audio
frame:

```python
def setup(proc):
    telephony_denoise.prewarm()

server = AgentServer(setup_fnc=setup)
```

A complete SIP agent is in `examples/sip_agent.py`; run it with
`python examples/sip_agent.py dev`.

### Echo cancellation needs both wires

Noise suppression works with the first wire alone. Echo cancellation does not.

Cancelling echo means subtracting the agent's voice from the caller's audio, and
to subtract it you need a copy of it. `EchoReferenceTap` supplies that copy. Skip
it and the canceller has nothing to work with, so callers keep hearing the agent
talk over itself.

The tap also fixes a timing problem. A whole TTS sentence can be handed to the
output in one burst, far ahead of when the caller actually hears it, so the tap
releases the reference at playback speed instead of arrival speed.

### On a plain AudioStream

Outside the agents framework the same filter works directly on a track:

```python
stream = rtc.AudioStream.from_track(track=track, noise_cancellation=denoiser)
```

Or drive it yourself, frame by frame:

```python
denoiser.prepare(16000)                 # optional, builds the filter up front
clean = denoiser.process(frame)         # one frame in, one frame out
denoiser.push_render_frame(agent_audio) # the echo reference
denoiser.close()
```

## Why it is language independent

Nothing in the pipeline recognises words. Stages work on the signal, not its
meaning:

| Stage | Backend | What it removes |
| --- | --- | --- |
| High-pass filter | WebRTC | Rumble, mains hum, handling noise |
| Echo canceller | WebRTC AEC3 | The agent's own voice bouncing back |
| Gain control | WebRTC AGC | Callers too close to or far from the mic |
| Noise suppressor | **DeepFilterNet3 (default)** | Hiss, fans, gym, café, babble, keyboards |

DeepFilterNet is a speech-enhancement model, not a speech recogniser: it does
not consult a lexicon or a language ID, so a Hindi caller, a Japanese caller and
a Portuguese caller get the same treatment. Only the STT downstream cares about
language.

For minimum latency you can fall back to classical WebRTC NS with
`DenoiseOptions(enhancer="webrtc")`.

## Tuning

Defaults are chosen for phone calls where you do not know the caller's
environment.

**Enhancer.** `enhancer="deepfilter"` (default) is the generic choice. Use
`enhancer="webrtc"` only when you need the lowest latency and your callers are
mostly on quiet lines.

**Echo delay** is the other setting that genuinely matters. `stream_delay_ms`
tells the canceller how long the agent's voice takes to return. On SIP that path
runs through an SFU, a gateway, the PSTN and a handset. Start at 120 ms, then
tune it against recordings from your actual carrier route.

## Tests

```bash
pytest                 # everything, about a minute
pytest -m "not slow"   # skip the real-time simulations
```

No LiveKit credentials or phone line needed; everything runs offline. The
DeepFilterNet weights are fetched once on first use, and tests that need them
skip rather than fail if the machine is offline.

| Test | Covers |
| --- | --- |
| `test_options_and_processor.py` | Options validation, WebRTC processor, and format handling |
| `test_buffers_and_resample.py` | PCM helpers and streaming resampler contracts |

## Layout

```
livekit/plugins/telephony_denoise/
    processor.py        the denoiser, an rtc.FrameProcessor
    neural.py           DeepFilterNet3 (ONNX) noise suppressor
    _resample.py        low-latency streaming resampler to and from 48 kHz
    echo_reference.py   taps agent speech in as the echo reference
    _buffers.py         int16 PCM plumbing
examples/sip_agent.py   a SIP agent with everything wired up
tests/
```

## Things worth knowing

### Dependencies and licensing

The runtime dependencies are `livekit`, `livekit-agents`, `numpy`, and
`deepfilter-stream`. This package is MIT licensed. `deepfilter-stream`
and its bundled DeepFilterNet3 weights are dual-licensed under MIT or
Apache-2.0; their attribution is recorded in [NOTICE](NOTICE). The model is
downloaded by `deepfilter-stream` at runtime and is **not** redistributed in
this package.

**Do not stack cancellers.** If you already pass `noise_cancellation.BVC()` or
run a Krisp filter in your frontend, remove it. Two suppressors in series fight
each other and the result is worse than either alone. Internally this plugin
never runs WebRTC NS and DeepFilterNet at the same time.

**AGC is a trade-off, and it is on by default.** It works by riding the gain, so
on a noisy line it lifts the background back up between words. Measured on a gym
recording it raised the noise floor by 3 dB and swung its gain over a 16 dB
range, which is audible as pumping and sounds worse than leaving it off.

It stays on because the alternative is worse where it counts. Attenuating the
same call to imitate a caller holding the handset away, and running Silero, the
VAD in the LiveKit voice pipeline, over the result:

| Caller arrives at | Output level, AGC off | AGC on | VAD |
| --- | --- | --- | --- |
| Nominal | -22.5 dB | -19.1 dB | fine either way |
| -12 dB | -34.5 dB | -18.4 dB | fine either way |
| -20 dB | -42.5 dB | -20.9 dB | fine either way |
| -30 dB | -52.6 dB | -25.8 dB | **fragments without AGC** |

At -30 dB without AGC the VAD splits one turn into two and loses 1.7 s of
speech, which the agent experiences as a caller being cut off mid-sentence. With
AGC it holds. Pumping is a cosmetic complaint; dropping a quiet caller is not.

If your callers arrive at consistent levels, `auto_gain_control=False` sounds
cleaner. Judge it on your own recordings.

**One denoiser per call.** Each instance holds the echo path and noise estimate
for a single conversation. Build a new one per job; never share across calls.

**Audio format.** SIP telephony is mono, and the denoiser accepts mono input
only. It also needs a sample rate that is a whole number of samples per 10 ms
block, meaning any multiple of 100 Hz: 8000, 16000, 24000, 32000 and 48000 all
qualify. Anything else, including a multi-channel track, is logged once and
passed through unfiltered rather than failing the call.

**First-run model download.** DeepFilterNet3's ONNX weights (~13 MB) download on
first use and are cached under the platform cache directory. Call
`telephony_denoise.prewarm()` from a worker setup hook, or let
`lk agent build` bake them in via the plugin's `download_files` hook, so no call
pays for it.

**CPU.** DeepFilterNet3 on ONNX Runtime is the heavy stage: about 13% of one
core per stream, near enough the same at 8 kHz and 48 kHz because the model
always runs at 48 kHz internally. WebRTC AEC/HPF/AGC on top come to well under
1%, so `enhancer="webrtc"` costs about 0.6% of a core if you need call density
more than suppression quality.

**Latency.** Input-sample to output-sample delay, measured through the whole
chain with AEC enabled:

| Sample rate | DeepFilterNet3 | WebRTC NS |
| --- | --- | --- |
| 8 kHz | 70 ms | 23 ms |
| 16 kHz | 62 ms | 20 ms |
| 24 kHz | 62 ms | - |
| 48 kHz | 58 ms | - |

Most of that is the model's own 32 ms lookahead, which is fixed. The resample to
48 kHz and back adds only a few milliseconds; a general-purpose resampler at a
comparable quality setting cost 140 ms at 8 kHz on its own, which is why
`_resample.py` exists. On a SIP call that already carries substantial network and
jitter delay, this is usually acceptable.

**If you are on LiveKit Cloud** and would rather not run any of this, one line
gets you Krisp's telephony model:

```python
noise_cancellation=noise_cancellation.BVCTelephony()   # billed per minute
```

This plugin exists for the case where you want that job done in your own
process, on your own hardware, at no per-minute cost.
