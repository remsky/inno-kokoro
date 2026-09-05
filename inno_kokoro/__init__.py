"""Kokoro-82M reference tuner. Use inno_kokoro.enroll (Tuner, enroll, fetch_weights, read) and inno_kokoro.prosody (stats, rate, tilt)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("inno-kokoro")
except PackageNotFoundError:  # running from a checkout
    __version__ = "0.0.0"
