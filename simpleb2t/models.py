"""Independent CNN + residual MLP and the joint-decoding controls."""

import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim=1024, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class ClinicalCNN(nn.Module):
    def __init__(self, cnn, dim=1024, depth=4, hidden=2048):
        super().__init__()
        self.cnn = cnn
        self.head = nn.Sequential(
            *[ResidualBlock(dim, hidden) for _ in range(depth)], nn.LayerNorm(dim)
        )

    def forward(self, x, positions):
        subjects = torch.zeros(len(x), dtype=torch.long, device=x.device)
        y = F.normalize(self.cnn(x, subjects, positions), dim=-1)
        return F.normalize(self.head(y), dim=-1)


class SentenceCNN(nn.Module):
    def __init__(self, cnn, transformer):
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer

    def contextualize(self, embeddings, lengths):
        assert sum(lengths) == len(embeddings) and min(lengths) > 0
        padded = embeddings.new_zeros(len(lengths), max(lengths), embeddings.shape[-1])
        mask = torch.zeros(padded.shape[:2], device=embeddings.device, dtype=torch.bool)
        offset = 0
        for row, n in enumerate(lengths):
            padded[row, :n] = embeddings[offset : offset + n]
            mask[row, :n] = True
            offset += n
        contextual = self.transformer(padded, mask=mask)
        return F.normalize(torch.cat([contextual[i, :n] for i, n in enumerate(lengths)]), dim=-1)

    def forward(self, x, positions, lengths):
        subjects = torch.zeros(len(x), device=x.device, dtype=torch.long)
        cnn = F.normalize(self.cnn(x, subjects, positions.expand(len(x), -1, -1)), dim=-1)
        return self.contextualize(cnn, lengths)


def batches(sequences, budget=128):
    batch = []
    size = 0
    for sequence in sequences:
        if batch and size + len(sequence) > budget:
            yield batch
            batch = []
            size = 0
        batch.append(sequence)
        size += len(sequence)
    if batch:
        yield batch


class ControlledModel(nn.Module):
    def __init__(self, core, mode):
        super().__init__()
        self.core = core
        self.mode = mode

    def forward(self, x, positions, lengths):
        if self.mode == "single_word":
            return self.core(x, positions, [1] * len(x))
        if self.mode in ("shared_pulses", "independent_pulses"):
            from .synthetic import pulse_windows

            windows = []
            offset = 0
            # Input contains onset and stable source-sentence ID, never MEG/text.
            values = x.detach().cpu().tolist()
            for n in lengths:
                rows = values[offset : offset + n]
                offset += n
                assert len({r[1] for r in rows}) == 1
                if self.training:
                    seed = int(torch.randint(0, 2**62, ()).item())
                else:
                    seed = int.from_bytes(
                        hashlib.sha256(
                            ("clinical-natural-pulses-v1/" + str(int(rows[0][1]))).encode()
                        ).digest()[:8],
                        "little",
                    ) % (2**63 - 1)
                generator = torch.Generator(device=x.device).manual_seed(seed)
                windows.append(
                    pulse_windows(
                        [r[0] for r in rows], 306, self.mode == "shared_pulses", generator, x.device
                    )
                )
            x = torch.cat(windows)
        return self.core(x, positions, lengths)


class TimingModel(SentenceCNN):
    def __init__(self, transformer, statistics, dim=1024):
        nn.Module.__init__(self)
        self.transformer = transformer
        self.encoder = nn.Sequential(nn.Linear(1, 128), nn.GELU(), nn.Linear(128, dim))
        self.register_buffer("log_mean", torch.tensor(statistics["log_mean"]))
        self.register_buffer("log_std", torch.tensor(statistics["log_std"]))

    def forward(self, intervals, positions, lengths):
        available = torch.isfinite(intervals)
        safe = torch.where(available, intervals, torch.ones_like(intervals))
        z = torch.where(
            available, (safe.log() - self.log_mean) / self.log_std, torch.zeros_like(intervals)
        )
        return self.contextualize(F.normalize(self.encoder(z[:, None]), dim=-1), lengths)
