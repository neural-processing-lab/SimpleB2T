"""Read public HDF5 recordings; preprocess locally; use immutable occurrence IDs."""

import collections
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from .io import DATA, bundled, read, write, sha


@lru_cache(None)
def natural():
    return bundled("natural.json.gz")


@lru_cache(None)
def groups():
    return bundled("groups.json.gz")


@lru_cache(None)
def records():
    return bundled("recordings.json.gz")


def published_intervals(frame):
    """Next published word, not next eligible word; no crossing sentence boundaries."""
    rows = list(frame.loc[frame.kind == "word"].sort_values("timeds", kind="stable").iterrows())
    result = {}
    for i, (index, row) in enumerate(rows):
        gap = None
        if i + 1 < len(rows):
            following = rows[i + 1][1]
            same = all(row[k] == following[k] for k in ("wavile", "sentenceidx"))
            delta = float(following.timeds) - float(row.timeds)
            if same and math.isfinite(delta) and delta > 0:
                gap = min(delta, 3.0)
        result[int(index)] = gap
    return result


def download(root, cross_subject=False, annotations_only=False):
    """Download only the public files used in the paper, at the pinned release."""
    from huggingface_hub import hf_hub_download

    for r in records():
        if r["subject"] != "0" and not cross_subject:
            continue
        for name in (("events",) if annotations_only else ("neural", "events")):
            entry = bundled("downloads.json")[r[name]]
            path = hf_hub_download(
                repo_id=entry["repo"],
                repo_type="dataset",
                revision=entry["revision"],
                filename=r[name],
                local_dir=root,
            )
            if entry["sha256"] is not None and sha(path) != entry["sha256"]:
                raise ValueError(f"Download checksum mismatch: {r[name]}")


def prepare(root, work, cross_subject=False, annotations_only=False):
    """One recording at a time, atomic cached arrays; rerunning skips completed work."""
    import pandas as pd

    work = Path(work)
    root = Path(root)
    for rid, r in enumerate(records()):
        if r["subject"] != "0" and not cross_subject:
            continue
        folder = work / "recordings" / str(rid)
        table = root / r["events"]
        if sha(table) != r["events_sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {table}")
        frame = pd.read_csv(table, sep="\t")
        if not (folder / "annotations.json").exists():
            intervals = published_intervals(frame)
            sentence_ids = {
                int(i): [str(row.wavile), str(row.sentenceidx)]
                for i, row in frame.loc[frame.kind == "word"].iterrows()
            }
            write(folder / "annotations.json", dict(intervals=intervals, sentence_ids=sentence_ids))
        if annotations_only or (folder / "receipt.json").exists():
            continue
        import h5py
        import mne
        from .normalization import recording_scale, word_window

        def attrs(v):
            if isinstance(v, bytes):
                v = v.decode()
            if isinstance(v, str):
                return [item.strip() for item in v.split(",")]
            return [x.decode() if isinstance(x, bytes) else str(x) for x in v]

        path = root / r["neural"]
        print(f"Preprocessing {rid + 1}/{len(records())}: {path.name}", flush=True)
        with h5py.File(path, "r") as f:
            names = attrs(f.attrs["channel_names"])
            kinds = attrs(f.attrs["channel_types"])
            rate = float(f.attrs["sample_frequency"])
            times = np.asarray(f["times"])
            assert len(names) == 306 and float(times[0]) == r["origin"]
            expected = float(times[0]) + np.arange(len(times)) / rate
            tolerance = max(
                1e-6, float(np.finfo(times.dtype).eps) * max(1.0, float(abs(times).max()))
            )
            assert np.allclose(times, expected, atol=tolerance, rtol=0)
            raw = mne.io.RawArray(
                np.asarray(f["data"], dtype=np.float64),
                mne.create_info(names, rate, kinds),
                verbose="ERROR",
            )
        raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False, exclude=[]))
        layout = mne.channels.read_layout("Vectorview-all")
        lookup = {n.replace(" ", ""): i for i, n in enumerate(layout.names)}
        xy = layout.pos[[lookup[n] for n in raw.ch_names], :2]
        positions = ((xy - xy.min(0)) / np.ptp(xy, axis=0)).astype("float32")
        np.testing.assert_array_equal(positions, np.load(DATA / "positions.npy"))
        raw.load_data().filter(0.1, 40.0, n_jobs=1, verbose="ERROR").resample(
            50.0, n_jobs=1, verbose="ERROR"
        )
        scaled, _ = recording_scale(raw.get_data())
        raw.close()
        del raw
        events = sorted(r["words"], key=lambda e: (e["start"], e["event_index"]))
        temporary = folder / "windows.tmp.npy"
        a = np.lib.format.open_memmap(
            temporary, mode="w+", dtype="float32", shape=(len(events), 306, 150)
        )
        for j, e in enumerate(events):
            a[j] = word_window(scaled, round((e["start"] - r["origin"]) * 50)).numpy()
        a.flush()
        del a, scaled
        temporary.replace(folder / "windows.npy")
        write(
            folder / "receipt.json",
            dict(
                events=[e["event_index"] for e in events],
                annotation_sha256=r["events_sha256"],
                source_bytes=path.stat().st_size,
                channels=names,
                shape=[len(events), 306, 150],
                preprocessing="filter .1-40 Hz; resample 50 Hz; recording RobustScaler; float32; baseline 25 samples; clip [-5,5]",
            ),
        )


class Windows:
    """Memory-mapped recording shards; indices always refer to fixed natural metadata."""

    def __init__(self, work, split, mode="ours", metadata=None):
        self.work = Path(work)
        self.mode = mode
        d = natural()[split] if metadata is None else metadata
        self.items, self.sequences = d["items"], d["sequences"]
        self.arrays = {}
        self.rows = {}
        self.annotations = {}
        if mode == "timing":
            for rid in {i["record"] for i in self.items}:
                self.annotations[rid] = read(
                    self.work / "recordings" / str(rid) / "annotations.json"
                )["intervals"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, indices):
        items = [self.items[int(i)] for i in indices]
        if self.mode in ("shared_pulses", "independent_pulses"):
            return np.asarray(
                [[i["onset"], i["synthetic_sequence_id"]] for i in items], dtype="float64"
            )
        if self.mode == "timing":
            return np.asarray(
                [self.annotations[i["record"]][str(i["event"])] for i in items], dtype="float32"
            )
        out = []
        for item in items:
            rid = item["record"]
            if rid not in self.arrays:
                folder = self.work / "recordings" / str(rid)
                self.arrays[rid] = np.load(folder / "windows.npy", mmap_mode="r")
                self.rows[rid] = {
                    e: j for j, e in enumerate(read(folder / "receipt.json")["events"])
                }
            out.append(self.arrays[rid][self.rows[rid][item["event"]]])
        return np.stack(out)


def cross_subject_metadata(work, subject):
    rid, r = next((i, r) for i, r in enumerate(records()) if r["subject"] == str(subject))
    annotation = read(Path(work) / "recordings" / str(rid) / "annotations.json")
    items = []
    sequences = collections.OrderedDict()
    for e in sorted(r["words"], key=lambda e: (e["start"], e["event_index"])):
        key = tuple(annotation["sentence_ids"][str(e["event_index"])])
        sequences.setdefault(key, []).append(len(items))
        items.append(
            dict(
                word=e["text"],
                record=rid,
                event=e["event_index"],
                onset=e["start"] - r["origin"],
                duration=e["duration"],
            )
        )
    for j, seq in enumerate(sequences.values()):
        for i in seq:
            items[i]["synthetic_sequence_id"] = 1000000000 + subject * 1000000 + j
    return dict(items=items, sequences=list(sequences.values()))


def bank(split):
    from .stitching import StitchBank

    items = natural()[split]["items"]
    return StitchBank(
        [i["word"] for i in items],
        [i["record"] for i in items],
        [round(i["onset"] * 50) for i in items],
    )
