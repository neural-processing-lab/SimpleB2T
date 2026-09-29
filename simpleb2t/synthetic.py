"""Stationary random pulse fields independent of stimulus/word events."""

import math
import torch
import torch.nn.functional as F

RATE_HZ = 3.0
SIGMAS = (3.0, 6.0, 12.0)  # 60, 120, 240 ms at 50 Hz


def pulse_stream(batch, channels, length, generator, device="cpu"):
    # Common padding exceeds every kernel radius: no special window-edge cues.
    padding = 48
    n = length + 2 * padding
    result = torch.zeros(batch, channels, n, device=device)
    variance = 0.0
    for sigma in SIGMAS:
        radius = int(4 * sigma)
        t = torch.arange(-radius, radius + 1, device=device)
        kernel = torch.exp(-0.5 * (t / sigma).square())
        rate = RATE_HZ / 50 / len(SIGMAS)
        # A Poisson process on the sample grid, thinned by pulse width.
        counts = torch.poisson(torch.full((batch, 1, n), rate, device=device), generator=generator)
        amplitudes = (
            torch.randn(batch, channels, n, generator=generator, device=device) * counts.sqrt()
        )
        smoothed = F.conv1d(
            amplitudes.reshape(batch * channels, 1, n), kernel[None, None], padding=radius
        )
        result += smoothed.reshape(batch, channels, n)
        variance += rate * float(kernel.square().sum())
    # Fixed population scale, never per-window or per-sentence normalization.
    return result[:, :, padding : padding + length] / math.sqrt(variance)


def pulse_windows(onsets, channels, shared, generator, device="cpu", length=150):
    starts = [round(float(t) * 50) for t in onsets]
    starts = [s - min(starts) for s in starts]
    if shared:
        stream = pulse_stream(1, channels, max(starts) + length, generator, device)[0]
        windows = torch.stack([stream[:, s : s + length] for s in starts])
    else:
        windows = pulse_stream(len(starts), channels, length, generator, device)
    windows = windows - windows[:, :, :25].mean(-1, keepdim=True)
    return windows.clamp(-5, 5)
