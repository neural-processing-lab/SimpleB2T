"""Same-word donor permutations; every sentence has disjoint three-second windows."""

import collections, hashlib
import numpy as np

DATA_SEED = 20260920


class StitchBank:
    def __init__(self, words, recordings, starts):
        self.words = list(words)
        self.recordings = np.asarray(recordings)
        self.starts = np.asarray(starts)
        pools = collections.defaultdict(list)
        for i, w in enumerate(words):
            pools[w].append(i)
        self.pools = {w: np.asarray(v, dtype=np.int64) for w, v in pools.items()}

    def valid(self, indices):
        if len(indices) != len(set(indices)):
            return False
        by = collections.defaultdict(list)
        for i in indices:
            by[int(self.recordings[i])].append(int(self.starts[i]))
        return all(all(b - a >= 150 for a, b in zip(sorted(v), sorted(v)[1:])) for v in by.values())

    def compatible(self, i, others):
        return all(
            self.recordings[i] != self.recordings[q]
            or abs(int(self.starts[i]) - int(self.starts[q])) >= 150
            for q in others
        )

    def assign(self, words, rng, forbidden=None, attempts=8):
        # Assign the rarest words first to avoid blocking scarce source windows.
        order = sorted(range(len(words)), key=lambda j: (len(self.pools[words[j]]), j))
        for _ in range(attempts):
            chosen = []
            out = [None] * len(words)
            used_records = set()
            for j in order:
                pool = self.pools[words[j]]
                candidates = (
                    rng.permutation(pool)
                    if len(pool) <= 48
                    else pool[rng.choice(len(pool), 48, replace=False)]
                )
                fallback = None
                pick = None
                excluded = set() if forbidden is None else forbidden[j]
                for raw in candidates:
                    i = int(raw)
                    if i in excluded or i in chosen:
                        continue
                    if any(
                        self.recordings[i] == self.recordings[q]
                        and abs(int(self.starts[i]) - int(self.starts[q])) < 150
                        for q in chosen
                    ):
                        continue
                    if self.recordings[i] not in used_records:
                        pick = i
                        break
                    if fallback is None:
                        fallback = i
                if pick is None:
                    pick = fallback
                if pick is None:
                    break
                out[j] = pick
                chosen.append(pick)
                used_records.add(self.recordings[pick])
            if None not in out:
                assert self.valid(out)
                assert [self.words[i] for i in out] == list(words)
                return out
        return None


def permutation_attempt(bank, sequences, rng):
    """Shuffle same-word donors one-to-one, then repair overlaps by safe swaps."""
    lengths = np.asarray([len(s) for s in sequences])
    offsets = np.r_[0, np.cumsum(lengths)]
    original = np.asarray([i for seq in sequences for i in seq], dtype=np.int64)
    assert len(original) == len(set(original.tolist()))
    owners = np.repeat(np.arange(len(sequences)), lengths)
    byword = collections.defaultdict(list)
    for j, i in enumerate(original):
        byword[bank.words[i]].append(j)
    byword = {w: np.asarray(v, dtype=np.int64) for w, v in byword.items()}
    assigned = original.copy()
    for positions in byword.values():
        assigned[positions] = rng.permutation(original[positions])
    bad = [
        s
        for s in range(len(sequences))
        if not bank.valid(assigned[offsets[s] : offsets[s + 1]].tolist())
    ]
    for _ in range(4):
        if not bad:
            break
        for s in rng.permutation(bad):
            lo, hi = offsets[s : s + 2]
            for j in range(lo, hi):
                old = int(assigned[j])
                others = [int(v) for q, v in enumerate(assigned[lo:hi], int(lo)) if q != j]
                if bank.compatible(old, others):
                    continue
                pool = byword[bank.words[old]]
                choices = (
                    rng.permutation(pool)
                    if len(pool) <= 128
                    else pool[rng.choice(len(pool), 128, replace=False)]
                )
                for partner in choices:
                    other = int(owners[partner])
                    if other == s:
                        continue
                    new = int(assigned[partner])
                    if not bank.compatible(new, others):
                        continue
                    a, b = offsets[other : other + 2]
                    other_values = [
                        int(v) for q, v in enumerate(assigned[a:b], int(a)) if q != partner
                    ]
                    if not bank.compatible(old, other_values):
                        continue
                    assigned[j], assigned[partner] = new, old
                    break
        bad = [s for s in bad if not bank.valid(assigned[offsets[s] : offsets[s + 1]].tolist())]
    assert np.array_equal(np.sort(original), np.sort(assigned))
    assert all(bank.words[a] == bank.words[b] for a, b in zip(original, assigned))
    return [assigned[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])], bad


def schedule(bank, templates, fallback, epoch):
    # Start from a feasible one-to-one assignment; accept only swaps that keep
    # both affected sentences non-overlapping. Rare fixed points are permitted.
    rng = np.random.default_rng(DATA_SEED + 10000 + epoch)
    lengths = np.asarray([len(s) for s in fallback])
    offsets = np.r_[0, np.cumsum(lengths)]
    assigned = np.asarray([i for seq in fallback for i in seq], dtype=np.int64)
    original = assigned.copy()
    owners = np.repeat(np.arange(len(fallback)), lengths)
    byword = collections.defaultdict(list)
    for j, i in enumerate(assigned):
        byword[bank.words[i]].append(j)
    for pool in byword.values():
        if len(pool) < 2:
            continue
        for _ in range(2):
            order = rng.permutation(pool)
            for j, partner in zip(order, np.roll(order, 1)):
                s = int(owners[j])
                other = int(owners[partner])
                old, new = int(assigned[j]), int(assigned[partner])
                if s == other:
                    assigned[j], assigned[partner] = new, old
                    continue
                a, b = offsets[s : s + 2]
                others = [int(v) for q, v in enumerate(assigned[a:b], int(a)) if q != j]
                if not bank.compatible(new, others):
                    continue
                a, b = offsets[other : other + 2]
                others = [int(v) for q, v in enumerate(assigned[a:b], int(a)) if q != partner]
                if not bank.compatible(old, others):
                    continue
                assigned[j], assigned[partner] = new, old
    assert np.array_equal(np.sort(original), np.sort(assigned))
    result = [assigned[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    assert all(bank.valid(seq) for seq in result)
    assert all([bank.words[i] for i in seq] == text for seq, text in zip(result, templates))
    flat = assigned.astype(np.int32)
    return result, hashlib.sha256(flat.tobytes()).hexdigest(), 0
