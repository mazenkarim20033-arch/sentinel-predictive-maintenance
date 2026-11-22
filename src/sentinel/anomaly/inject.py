"""Controlled synthetic sensor anomalies. Magnitudes are in units of the healthy per-signal standard deviation.
Results on these are synthetic-fault results and must never be reported as accuracy on real anomalies."""
import numpy as np

KINDS = ("spike", "step", "ramp", "stuck", "noise_burst")


def inject(
    z: np.ndarray, kind: str, magnitude: float, signal: int, start: int, duration: int, scale: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (modified copy of z, boolean mask of anomalous steps). Operates on one asset's sequence (steps, signals)."""
    if kind not in KINDS:
        raise ValueError(f"unknown anomaly kind {kind!r}")
    out = z.copy()
    n = len(z)
    duration = 1 if kind == "spike" else duration
    end = min(start + duration, n)
    if start < 0 or start >= n:
        raise ValueError("start outside the sequence")
    amp = magnitude * scale[signal]
    seg = slice(start, end)
    if kind == "spike":
        out[start, signal] += rng.choice([-1.0, 1.0]) * amp
    elif kind == "step":
        out[seg, signal] += amp
    elif kind == "ramp":
        out[seg, signal] += np.linspace(0, amp, end - start)
    elif kind == "stuck":
        out[seg, signal] = z[start, signal]
    else:
        out[seg, signal] += rng.normal(0.0, amp, end - start)
    mask = np.zeros(n, dtype=bool)
    mask[seg] = True
    return out, mask
