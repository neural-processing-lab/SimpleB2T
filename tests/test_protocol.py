"""Small CPU tests of the scientific protocol, not just implementation shapes."""

import itertools
import numpy as np
import pandas as pd
import torch
import pytest
from simpleb2t.metrics import metrics, word_edit_distance
from simpleb2t.lm import nbest
from simpleb2t.evaluation import word_scores, summarize_rows
from simpleb2t.data import published_intervals
from simpleb2t.stitching import StitchBank
from simpleb2t.normalization import word_window


def test_wer_uses_edit_alignment_not_position_error():
    assert word_edit_distance("a b c d".split(), "b c d e".split()) == 2
    r = summarize_rows([dict(reference="a b c d", prediction="b c d e")])
    assert r["word_error_rate"] == 0.5
    assert r["position_error_rate"] == 1.0


def test_macro_accuracy_and_replica_denominator():
    assert metrics(["a", "a", "a", "b"], ["a", "a", "a", "a"])["top1_balanced_accuracy"] == 0.5
    rows = [dict(reference="a", prediction="a" if i < 3 else "b") for i in range(500)]
    assert summarize_rows(rows)["sentence_exact_accuracy"] == 0.006


def test_all_observation_subsets():
    assert [len(list(itertools.combinations(range(5), k))) for k in range(1, 6)] == [
        5,
        10,
        10,
        5,
        1,
    ]


def test_single_observation_evidence_weight_equivalence():
    torch.manual_seed(1)
    v = torch.nn.functional.normalize(torch.randn(3, 5, 8), dim=-1).numpy()
    targets = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    base = word_scores(v, (2,), 0.04, 0.0, targets)
    combined = word_scores(v, (2,), 0.04, 2.0, targets)
    np.testing.assert_allclose(combined, 3 * base, atol=1e-5)


class ToyLM:
    """Its probabilities depend only on generated prefixes, never a reference."""

    def score_words(self, prefixes):
        return np.asarray(
            [[np.log(0.6), np.log(0.3)] if not p else [np.log(0.2), np.log(0.7)] for p in prefixes]
        )

    def score_end(self, prefixes):
        return np.asarray([-0.1 if p[-1] == 0 else -0.4 for p in prefixes])


def test_beam_matches_exhaustive_search_and_terminal_score():
    evidence = np.array([[0.1, 0.3], [0.4, 0.2]])
    lm = ToyLM()
    expected = []
    for candidate in itertools.product(range(2), repeat=2):
        s = sum(
            evidence[j, w] + 0.5 * lm.score_words([candidate[:j]])[0, w]
            for j, w in enumerate(candidate)
        )
        s += 0.5 * lm.score_end([candidate])[0]
        expected.append((candidate, s))
    expected.sort(key=lambda p: -p[1])
    actual = nbest(evidence, lm, 0.5, 4)
    assert [p[0] for p in actual] == [p[0] for p in expected]
    np.testing.assert_allclose([p[1] for p in actual], [p[1] for p in expected])


def test_timing_does_not_cross_sentence_and_caps_unrecoverable_intervals():
    frame = pd.DataFrame(
        dict(
            kind=["word"] * 4,
            timeds=[0.0, 0.2, 4.0, 5.0],
            wavile=["x"] * 4,
            sentenceidx=[1, 1, 1, 2],
        )
    )
    assert published_intervals(frame) == {0: 0.2, 1: 3.0, 2: None, 3: None}


def test_stitching_disallows_overlap_and_reused_occurrences():
    bank = StitchBank(["a"] * 4, [0, 0, 0, 1], [0, 100, 150, 0])
    assert not bank.valid([0, 1])
    assert not bank.valid([0, 0])
    assert bank.valid([0, 2]) and bank.valid([0, 3])


def test_preprocessing_order_and_no_input_mutation():
    x = np.arange(400, dtype=float).reshape(2, 200) / 10
    before = x.copy()
    out = word_window(x, 1)
    expected = torch.tensor(x[:, 1:151], dtype=torch.float32)
    expected = (expected - expected[:, :25].mean(-1, keepdim=True)).clamp(-5, 5)
    torch.testing.assert_close(out, expected)
    np.testing.assert_array_equal(x, before)


def test_shuffled_control_changes_assignments_without_changing_evidence_pool():
    vectors = np.arange(595 * 5 * 2).reshape(595, 5, 2)
    permutation = np.random.default_rng(20260919).permutation(595 * 5)
    shuffled = vectors.reshape(-1, 2)[permutation].reshape(595, 5, 2)
    assert not np.array_equal(vectors, shuffled)
    np.testing.assert_array_equal(
        np.sort(vectors.reshape(-1, 2), axis=0), np.sort(shuffled.reshape(-1, 2), axis=0)
    )


def test_summary_retains_partition_and_deduplicates_training_seeds(tmp_path):
    from simpleb2t.io import write
    from simpleb2t.experiments import aggregate

    settings = dict(model="ours", seed=0, split="test", k=5)
    scores = dict(
        word_error_rate=0.5,
        sentence_exact_accuracy=0.2,
        top1_balanced_accuracy=0.3,
        top1_word_accuracy=0.4,
    )
    for name in ["clinical", "duplicate_ablation"]:
        write(
            tmp_path / "decoding" / name / "result.json",
            dict(settings=settings, metrics={"Core": scores}),
        )
        write(tmp_path / "decoding" / name / "predictions.json", [])
    result = aggregate(tmp_path)
    assert len(result) == 1 and result[0]["partition"] == "Core"
    assert result[0]["seeds"] == [0] and result[0]["WER"]["mean"] == 50.0


def test_cnn_forward_backward_and_independence():
    from simpleb2t.training import build
    from simpleb2t.io import DATA

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = build("ours", {}, "cpu")
    positions = torch.from_numpy(np.load(DATA / "positions.npy"))[None].expand(2, -1, -1)
    x = torch.randn(2, 306, 150)
    model.eval()
    pred = model(x, positions)
    changed = x.clone()
    changed[1] = 10 * changed[1]
    torch.testing.assert_close(pred[0], model(changed, positions)[0])
    assert pred.shape == (2, 1024)
    torch.testing.assert_close(pred.norm(dim=-1), torch.ones(2))
    pred.sum().backward()
    assert model.head[0].net[1].weight.grad is not None


def test_contrastive_loss_duplicate_handling():
    from simpleb2t.vendor.loss import SigLipLoss

    loss = SigLipLoss(
        norm_kind="y",
        temperature=True,
        bias=True,
        identical_candidates_threshold=0.999,
        reweigh_positives=True,
    )
    prediction = torch.randn(3, 8, requires_grad=True)
    target = torch.randn(3, 8)
    target[1] = target[0]
    value = loss(prediction, target)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(prediction.grad).all()


def test_full_training_resume_restores_optimizer_and_rng(tmp_path, monkeypatch):
    """Interrupt after a saved epoch; resumption must equal uninterrupted training."""
    from simpleb2t import training as train
    from simpleb2t.io import read
    import copy

    config = {
        "cnn": {},
        "transformer": {},
        "runs": {
            "ours": {
                "0": {
                    "training": {
                        "loss": {
                            "norm_kind": "y",
                            "temperature": True,
                            "bias": True,
                            "identical_candidates_threshold": 0.999,
                            "reweigh_positives": True,
                        },
                        "lr": 0.001,
                        "max_epochs": 3,
                        "patience": 10,
                    },
                    "stop_after_epoch": 3,
                }
            }
        },
    }
    data = tmp_path / "data"
    data.mkdir()
    generator = np.random.default_rng(3)
    np.savez(
        data / "targets.npz",
        words=np.array(["a", "b"]),
        embeddings=generator.normal(size=(2, 1024)).astype("float32"),
    )
    np.save(data / "positions.npy", np.zeros((2, 2), dtype="float32"))
    monkeypatch.setattr(train, "DATA", data)
    monkeypatch.setattr(
        train,
        "bundled",
        lambda name: (
            copy.deepcopy(config) if name == "experiment.json" else {"clinical": ["a", "b"]}
        ),
    )
    group = [dict(word="a", indices=list(range(5))), dict(word="b", indices=list(range(5, 10)))]
    monkeypatch.setattr(train, "groups", lambda: {"train": group})

    class Array:
        def __init__(self, *args):
            self.items = [dict(word="a" if i < 5 else "b") for i in range(10)]
            self.values = np.arange(80, dtype="float32").reshape(10, 8) / 80

        def __getitem__(self, ids):
            return self.values[ids]

    monkeypatch.setattr(train, "Windows", Array)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(8, 1024))

        def forward(self, x, positions):
            return torch.nn.functional.normalize(self.layers(x), dim=-1)

    monkeypatch.setattr(train, "build", lambda *args: Tiny())

    def validation(model, kind, array, device, positions):
        model.eval()
        with torch.no_grad():
            v = model(torch.from_numpy(array[list(range(10))]), positions).reshape(2, 5, 1024)
        return v, ["a", "b"], [list(range(5)), list(range(5, 10))]

    monkeypatch.setattr(train, "validation_vectors", validation)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train.run(full, "ours", 0, "cpu")
    original_save = train.atomic_torch

    def interrupt(path, value):
        original_save(path, value)
        if path.name == "last.pt" and value["epoch"] == 1:
            raise InterruptedError("simulated interruption")

    monkeypatch.setattr(train, "atomic_torch", interrupt)
    with pytest.raises(InterruptedError):
        train.run(resumed, "ours", 0, "cpu")
    monkeypatch.setattr(train, "atomic_torch", original_save)
    train.run(resumed, "ours", 0, "cpu")
    a = torch.load(full / "models/ours/0/last.pt", weights_only=False)
    b = torch.load(resumed / "models/ours/0/last.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert read(full / "models/ours/0/history.json") == read(resumed / "models/ours/0/history.json")


def test_decoder_outputs_partitions_and_five_single_observation_replicas(tmp_path, monkeypatch):
    from simpleb2t import evaluation as evaluate
    from simpleb2t.io import read

    words = [f"w{i}" for i in range(92)]
    groups = [
        dict(word="w0", sentence_id=1, word_position=1, indices=[0] * 5),
        dict(word="w1", sentence_id=101, word_position=1, indices=[1] * 5),
    ]
    monkeypatch.setattr(evaluate, "groups", lambda: {"test": groups})
    monkeypatch.setattr(
        evaluate,
        "bundled",
        lambda name: {"clinical": words} if name == "vocabularies.json" else {"temperature": 0.04},
    )
    monkeypatch.setattr(evaluate, "targets", lambda words: torch.eye(92))
    vectors = np.zeros((2, 5, 92), dtype="float32")
    vectors[0, :, 0] = 1
    vectors[1, :, 1] = 1
    source = tmp_path / "vectors.npy"
    np.save(source, vectors)
    monkeypatch.setattr(evaluate, "export", lambda *args, **kwargs: source)
    result = evaluate.decode(tmp_path, "ours", 0, k=1, weight=0, device="cpu", tag="test")
    assert set(result["metrics"]) == {"Core", "Expanded", "Full"}
    assert result["metrics"]["Full"]["word_error_rate"] == 0
    assert len(read(tmp_path / "decoding/test/predictions.json")) == 10
    with pytest.raises(AssertionError):
        evaluate.decode(tmp_path, "ours", 0, k=5, weight=0, device="cpu", tag="test")
