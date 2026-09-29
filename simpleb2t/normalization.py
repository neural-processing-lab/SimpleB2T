"""Recording-level scaling, then window baseline and clamp."""

import numpy as np


def recording_scale(data):
    from sklearn.preprocessing import RobustScaler

    # The reference preprocessing applies sklearn's
    # default RobustScaler independently to channels over the entire recording.
    scaler = RobustScaler()
    scaled = scaler.fit_transform(np.asarray(data, dtype=np.float64).T).T
    return scaled, {
        "method": "sklearn.preprocessing.RobustScaler defaults; channelwise over supplied recording/segment",
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "fit_scope": "this recording/segment only; fixed unsupervised preprocessing; no labels used",
    }


def word_window(data, start, length=150, baseline_samples=25, clamp=5.0):
    import torch

    # Match the original order: scaled float64 raw -> float32 window -> baseline
    # subtraction -> clamp. Do not recenter or rescale this window afterwards.
    x = torch.from_numpy(np.array(data[:, start : start + length], dtype=np.float32, copy=True))
    if x.shape[1] != length:
        raise ValueError("Incomplete word window")
    x -= x[:, :baseline_samples].mean(dim=-1, keepdim=True)
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite neural window")
    return x.clamp(-clamp, clamp)
