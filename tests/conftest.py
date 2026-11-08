import pytest

from inno_kokoro import pitch


@pytest.fixture(autouse=True)
def numpy_tracker(monkeypatch):
    """The suite tests the port even with the praat extra installed; test_head_stats_matches_praat opts into Praat."""
    monkeypatch.setattr(pitch, "PRAAT", False)
