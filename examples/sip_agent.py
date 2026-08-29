"""LiveKit SIP telephony agent with self-hosted noise and echo suppression.

The audio filtering is the point of this file; the speech models are ordinary
and meant to be swapped for whatever you already use.

Three wires do the work:

  1. The model is loaded in `setup`, once per worker process, so no call pays
     for the download and ONNX session build inside its first audio frame.

  2. The denoiser is handed to `AudioInputOptions.noise_cancellation`, so
     LiveKit runs it on every inbound frame from the caller before speech
     recognition sees it.

  3. The agent's own outgoing speech is tapped into the same denoiser, which is
     what lets it cancel the echo of its own voice coming back down the line.

Run it with:

    python examples/sip_agent.py dev

Inbound calls reach the agent through a SIP trunk and dispatch rule, which are
configured on the LiveKit side rather than here.
"""

from __future__ import annotations

import logging

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.plugins import telephony_denoise

logger = logging.getLogger("sip-agent")

# Narrowband phone audio, so there is no point paying for a wider pipeline.
INPUT_SAMPLE_RATE = 16000

INSTRUCTIONS = """You are a helpful voice assistant speaking with a caller on
the phone. Keep replies short and conversational. Reply in the language the
caller uses."""


def setup(proc: JobProcess) -> None:
    # Runs once per worker process, before any call is accepted.
    telephony_denoise.prewarm()


server = AgentServer(setup_fnc=setup)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    denoiser = telephony_denoise.TelephonyDenoiser(
        telephony_denoise.DenoiseOptions(
            # Cancels the agent's own voice echoing back off the caller's
            # handset or the carrier's hybrid.
            echo_cancellation=True,
            # Suppresses background noise via DeepFilterNet3 by default
            # (gym, café, babble, hiss — unknown caller environments).
            noise_suppression=True,
            # Removes rumble and mains hum below the voice band.
            high_pass_filter=True,
            # Evens out callers who are too close to or far from the handset.
            auto_gain_control=True,
            # Tune this to your typical SIP round-trip delay.
            stream_delay_ms=120,
        )
    )
    # Build the filter now rather than on the first frame from the caller, so
    # the greeting below can be used as an echo reference from its first word.
    denoiser.prepare(INPUT_SAMPLE_RATE)

    session = AgentSession(
        # Multilingual models keep the whole pipeline language-agnostic, to
        # match the filter. Swap these for your own plugins or providers.
        stt="deepgram/nova-3:multi",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-2",
    )

    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=denoiser,
                sample_rate=INPUT_SAMPLE_RATE,
                # The denoiser already applies gain control; running LiveKit's
                # as well would compress the signal twice.
                auto_gain_control=False,
            ),
        ),
    )

    # Must come after start(): RoomIO installs its own audio output during
    # start and would overwrite anything set beforehand.
    session.output.audio = telephony_denoise.EchoReferenceTap(
        denoiser, next_in_chain=session.output.audio
    )

    logger.info("call ready, audio filter active: %s", denoiser.enhancer)

    await session.generate_reply(
        instructions="Greet the caller briefly and ask how you can help."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(server)
