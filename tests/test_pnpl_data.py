"""PNPL must supply the same occurrences and numerical inputs as before."""
import hashlib
import json
import numpy as np
import pytest
import torch
from simpleb2t import data
from simpleb2t.io import bundled, write


def test_pnpl_metadata_matches_existing_experiments():
    # Recording IDs differ between packages. Everything must match after mapping
    # by filename, including sentence membership and synthetic-field seeds.
    # Fingerprints from the old release, retained without a duplicate dataset.
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest(data.natural()) == "54b7f02fd8fae66c68b003f1a4307a48bca1d3a6127c4d94e80f72ed2a54e12a"
    assert digest(data.groups()) == "35cd2d034055be1ad042172182ed769dcf2692fdbe22bbf7675aa15126c3782a"


@pytest.mark.parametrize("split", ["test", "dev"])
def test_clinical_batches_preserve_all_five_donors(tmp_path, monkeypatch, split):
    write(tmp_path / "data.json", dict(root=str(tmp_path), cache=str(tmp_path / "cache")))
    batch = data.ClinicalWindows(tmp_path, split)
    monkeypatch.setattr(batch.dataset.store, "window",
        lambda record, onset: torch.full((306, 150), float(onset)))
    # Include Core/Expanded and sentence boundaries; preserve repeats introduced
    # by padded inference batches, but do not change the benchmark assignment.
    rows = data.groups()[split]
    selected = [rows[0], rows[-1], rows[1], rows[0]]
    indices = [i for row in selected for i in row["indices"]]
    result = batch[indices]
    assert result.shape == (20, 306, 150)
    np.testing.assert_array_equal(result[:, 0, 0],
        np.asarray([batch.items[i]["onset"] for i in indices], dtype="float32"))
    # Verify every donor against the public dataset's metadata, without MEG I/O.
    for index, (sentence, position, member) in batch.locations.items():
        donor = batch.dataset.metadata(sentence)["occurrences"][position][member]
        old = batch.items[index]
        assert donor["neural"] == data.records()[old["record"]]["neural"]
        assert donor["event_index"] == old["event"]
    batch.close()


def test_natural_windows_use_pnpl_for_all_splits(tmp_path, monkeypatch):
    from pnpl.datasets.clinical_communication.recordings import RecordingStore
    write(tmp_path / "data.json", dict(root=str(tmp_path), cache=str(tmp_path / "cache")))
    monkeypatch.setattr(RecordingStore, "window",
        lambda self, record, onset: torch.full((306, 150), float(onset)))
    for split in ("train", "val", "test"):
        batch = data.Windows(tmp_path, split)
        indices = [0, len(batch) - 1, 10, 0]
        np.testing.assert_array_equal(batch[indices][:, 0, 0],
            np.asarray([batch.items[i]["onset"] for i in indices], dtype="float32"))
        batch.close()


def test_pnpl_preprocessing_matches_old_windows(tmp_path, monkeypatch):
    """Real PNPL HDF5 -> MNE -> cache -> window path, with a small fixture."""
    import h5py
    import mne
    from simpleb2t.io import DATA
    from sklearn.preprocessing import RobustScaler

    layout = mne.channels.read_layout("Vectorview-all")
    xy = layout.pos[:, :2]
    positions = ((xy - xy.min(0)) / np.ptp(xy, axis=0)).astype("float32")
    order = [np.where(np.all(positions == p, axis=1))[0][0] for p in np.load(DATA / "positions.npy")]
    names = [layout.names[i].replace(" ", "") for i in order]
    kinds = ["mag" if n.endswith("1") else "grad" for n in names]
    signal = np.random.default_rng(7).normal(size=(306, 5000))
    path = tmp_path / "recording.h5"
    with h5py.File(path, "w") as f:
        f["data"] = signal
        f["times"] = 17. + np.arange(5000) / 250
        f.attrs["channel_names"] = names
        f.attrs["channel_types"] = kinds
        f.attrs["sample_frequency"] = 250.
    record = dict(data.records()[0], rate=250., origin=17.)
    store = data._store(tmp_path, download=False)
    monkeypatch.setattr(store, "_source", lambda *args: path)
    actual = store.window(record, .51)
    raw = mne.io.RawArray(signal, mne.create_info(names, 250., kinds), verbose="ERROR")
    raw.filter(.1, 40., verbose="ERROR").resample(50., verbose="ERROR")
    scaled = RobustScaler().fit_transform(raw.get_data().T).T
    start = round(.51 * 50)
    expected = torch.tensor(scaled[:, start:start + 150], dtype=torch.float32)
    expected -= expected[:, :25].mean(-1, keepdim=True)
    torch.testing.assert_close(actual, expected.clamp(-5, 5), rtol=0, atol=0)
    store.close()
    # Reopen without source data: the continuous PNPL cache suffices.
    path.unlink()
    cached = data._store(tmp_path, download=False)
    torch.testing.assert_close(cached.window(record, .51), actual, rtol=0, atol=0)
    cached.close()


def test_timing_preparation_never_opens_meg(tmp_path, monkeypatch):
    import hashlib
    table = tmp_path / "events.tsv"
    table.write_text("kind\ttimeds\twavile\tsentenceidx\nword\t1\ta\t0\nword\t1.3\ta\t0\nword\t2\ta\t1\n")
    record = dict(data.records()[0], events_sha256=hashlib.sha256(table.read_bytes()).hexdigest())
    monkeypatch.setattr(data, "records", lambda: [record])
    store = data._store(tmp_path)
    def source(relative, checksum):
        assert relative == record['events']
        assert checksum == record['events_sha256']
        return table
    monkeypatch.setattr(store, "_source", source)
    monkeypatch.setattr(store, "array", lambda r: pytest.fail("Timing tried to load MEG"))
    monkeypatch.setattr(data, "_store", lambda *args, **kwargs: store)
    data.download(tmp_path, annotations_only=True)
    data.prepare(tmp_path, tmp_path / 'work', annotations_only=True)
    from simpleb2t.io import read
    intervals = read(tmp_path / 'work/recordings/0/annotations.json')['intervals']
    assert intervals['0'] == pytest.approx(.3)
    assert intervals['1'] is None and intervals['2'] is None
