"""The pitch gate picks the nearest packs."""

import numpy as np

from inno_kokoro.enroll import MIN_PACKS, blend_weights


class FakeTuner:
    gate = 4.0
    grade_pen = 1.0
    scale = np.array([1.0, 1.1, 0.8])
    # F0 mean st, F0 sd st, syllables/s, deliberately not sorted by pitch, like the real table
    stats = np.array([[30.0, 3.0, 4.0], [50.0, 3.0, 4.0], [10.0, 3.0, 4.0], [31.0, 3.0, 4.0], [32.0, 3.0, 4.0]])
    grades = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    names = ["p30", "p50", "p10", "p31", "p32"]


def test_pitch_gate():
    """Inside the gate, blend the eligible packs; outside it, fall back to the nearest MIN_PACKS."""
    t = FakeTuner()
    assert MIN_PACKS == 3, "these cases assume the 3-pack fallback"

    w = blend_weights(t, np.array([31.0, 3.0, 4.0]))  # inside the gate: only the 30/31/32 cluster is eligible
    assert set(w) <= {"p30", "p31", "p32"}, w
    assert abs(sum(w.values()) - 1) < 1e-6, w

    w = blend_weights(t, np.array([9.0, 3.0, 4.0]))  # below the range: nearest 3 are p10, p30, p31
    assert set(w) <= {"p10", "p30", "p31"}, w
    assert "p10" in w, w
    assert abs(sum(w.values()) - 1) < 1e-6, w

    w = blend_weights(t, np.array([60.0, 3.0, 4.0]))  # above the range: nearest 3 are p50, p32, p31
    assert set(w) <= {"p50", "p32", "p31"}, w
    assert "p50" in w, w
