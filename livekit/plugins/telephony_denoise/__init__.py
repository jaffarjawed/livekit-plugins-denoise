"""Self-hosted noise and echo suppression for LiveKit SIP telephony."""

from livekit.agents import Plugin

from .echo_reference import EchoReferenceTap
from .log import logger
from .neural import prewarm
from .processor import DenoiseOptions, Enhancer, TelephonyDenoiser
from .version import __version__

__all__ = [
    "DenoiseOptions",
    "EchoReferenceTap",
    "Enhancer",
    "TelephonyDenoiser",
    "__version__",
    "prewarm",
]


class TelephonyDenoisePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        # Fetches the DeepFilterNet3 weights so `lk agent build` bakes them into
        # the image; otherwise the first call of a fresh worker pays for the
        # download on the event loop, inside its first 10 ms frame.
        prewarm()


Plugin.register_plugin(TelephonyDenoisePlugin())
