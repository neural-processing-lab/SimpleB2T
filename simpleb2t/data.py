"""PNPL data loading, adapted to the model's flat word indices."""

import collections
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
from .io import DATA, bundled, read, write, sha


@lru_cache(None)
def natural():
    from pnpl.datasets.clinical_communication.manifest import load

    # Old analysis files use a larger recording list (including other subjects).
    # Match by filename, never by PNPL's internal recording number.
    lookup = {r["neural"]: i for i, r in enumerate(records())}
    record_ids = [lookup[r["neural"]] for r in load("recordings.json")]
    return {
        split: dict(items=[dict(item, record=record_ids[item["record"]])
                           for item in load(f"natural_{split}.json.gz")["items"]],
                    sequences=load(f"natural_{split}.json.gz")["sequences"])
        for split in ("train", "val", "test")
    }


@lru_cache(None)
def groups():
    from pnpl.datasets.clinical_communication.manifest import load
    return load("groups.json.gz")


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


def _store(root, cache_path=None, download=True):
    """PNPL's continuous cache, also accepting the other-subject recordings."""
    from pnpl.datasets import LibriBrain100
    from pnpl.datasets.clinical_communication.recordings import RecordingStore
    from pnpl.datasets.clinical_communication.manifest import load

    class Store(RecordingStore):
        # ClinicalCommunication has pinned metadata for subject 0. The optional
        # cross-subject experiment uses regular LibriBrain100 file access.
        def _source(self, relative, checksum):
            if relative in load("downloads.json"):
                return super()._source(relative, checksum)
            path = self.root / relative
            if self.download:
                LibriBrain100.ensure_file_download(str(path), str(self.root),
                    repo_id=bundled("downloads.json")[relative]["repo"])
            if not path.exists():
                raise FileNotFoundError(path)
            if checksum is not None and sha(path) != checksum:
                raise ValueError(f"Source checksum mismatch: {path}")
            return path

        def _identity(self, record):
            if record["neural"] in load("downloads.json"):
                return super()._identity(record)
            import mne
            import sklearn
            import torch
            from pnpl.datasets.clinical_communication.recordings import PIPELINE
            return dict(pipeline=PIPELINE, neural=record["neural"],
                        source_sha256=bundled("downloads.json")[record["neural"]]["sha256"],
                        events_sha256=record["events_sha256"], origin=record["origin"],
                        mne=mne.__version__, sklearn=sklearn.__version__,
                        numpy=np.__version__, torch=torch.__version__)

    return Store(root, cache_path=cache_path, download=download)


def clinical_dataset(root, partition="test", cache_path=None):
    """All five donors; observation averaging is done later by the model."""
    from pnpl.datasets import ClinicalCommunication
    return ClinicalCommunication(root, partition=partition, k=5,
                                 test_sentences=200, cache_path=cache_path)


def download(root, cross_subject=False, annotations_only=False):
    """Ask PNPL for source files. Timing controls download only event tables."""
    store = _store(root)
    for r in records():
        if r["subject"] != "0" and not cross_subject:
            continue
        for name in (("events",) if annotations_only else ("neural", "events")):
            checksum = r["events_sha256"] if name == "events" else bundled("downloads.json")[r[name]]["sha256"]
            store._source(r[name], checksum)


def prepare(root, work, cross_subject=False, annotations_only=False):
    """Prepare PNPL's continuous cache and the timing-only annotation index."""
    import pandas as pd

    work, root = Path(work), Path(root).expanduser().resolve()
    cache = root / ".clinical_cache"
    write(work / "data.json", dict(root=str(root), cache=str(cache)))
    store = _store(root, cache)
    try:
        for rid, r in enumerate(records()):
            if r["subject"] != "0" and not cross_subject:
                continue
            table = store._source(r["events"], r["events_sha256"])
            frame = pd.read_csv(table, sep="\t")
            sentence_ids = {
                int(i): [str(row.wavile), str(row.sentenceidx)]
                for i, row in frame.loc[frame.kind == "word"].iterrows()
            }
            write(work / "recordings" / str(rid) / "annotations.json",
                  dict(intervals=published_intervals(frame), sentence_ids=sentence_ids))
            if not annotations_only:
                print(f"Preparing {r['neural']} with PNPL", flush=True)
                store.array(r)
    finally:
        store.close()


class Windows:
    """Batches of PNPL windows; indices refer to the fixed natural word list."""

    def __init__(self, work, split, mode="ours", metadata=None):
        self.work = Path(work)
        self.mode = mode
        d = natural()[split] if metadata is None else metadata
        self.items, self.sequences = d["items"], d["sequences"]
        self.dataset = None
        self.store = None
        if mode not in ("timing", "shared_pulses", "independent_pulses"):
            config = read(self.work / "data.json")
            if metadata is None and split in ("train", "val"):
                self.dataset = clinical_dataset(config["root"], split, config["cache"])
                self.store = self.dataset.store
            else:
                self.store = _store(config["root"], config["cache"])
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
        if self.dataset is not None:
            return np.stack([self.dataset[int(i)]["meg"].numpy() for i in indices])
        return np.stack([self.store.window(records()[i["record"]], i["onset"]).numpy()
                         for i in items])

    def close(self):
        if self.store is not None:
            self.store.close()


class ClinicalWindows:
    """Adapt PNPL clinical sentences to the decoder's donor-index batches.

    Keep only the last sentence in memory. PNPL owns window extraction and the
    five-occurrence assignment; the decoder owns embedding aggregation.
    """

    def __init__(self, work, split):
        config = read(Path(work) / "data.json")
        self.dataset = clinical_dataset(config["root"], split, config["cache"])
        self.items = natural()["val" if split == "dev" else "test"]["items"]
        self.locations = {}
        self.cached = None
        self.cached_id = None
        sentences = collections.defaultdict(list)
        for group in groups()[split]:
            sentences[group["sentence_id"]].append(group)
        for sentence, rows in enumerate(sentences.values()):
            for position, row in enumerate(rows):
                for member, index in enumerate(row["indices"]):
                    self.locations[index] = (sentence, position, member)

    def __getitem__(self, indices):
        out = []
        for index in indices:
            sentence, position, member = self.locations[int(index)]
            if sentence != self.cached_id:
                self.cached = self.dataset[sentence]["meg"].numpy()
                self.cached_id = sentence
            out.append(self.cached[position, member])
        return np.stack(out)

    def close(self):
        self.dataset.close()
        self.cached = None


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
