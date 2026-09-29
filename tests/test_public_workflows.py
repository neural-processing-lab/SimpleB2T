"""Guard the public entry points against silent protocol changes."""
import numpy as np
import pytest
import torch
from simpleb2t.evaluation import word_scores
from simpleb2t.workflows import seed_list, ablation_plan


def test_evidence_ablations_preserve_their_scale():
    # Individual-only is mean log probability, without the factor alpha.
    generator = torch.Generator().manual_seed(9)
    vectors = torch.nn.functional.normalize(torch.randn(7, 5, 8, generator=generator), dim=-1).numpy()
    targets = torch.nn.functional.normalize(torch.randn(4, 8, generator=generator), dim=-1)
    embedding = word_scores(vectors, range(5), .04, 2, targets, evidence="embedding")
    individual = word_scores(vectors, range(5), .04, 2, targets, evidence="individual")
    combined = word_scores(vectors, range(5), .04, 2, targets)
    np.testing.assert_allclose(combined, embedding + 2 * individual)
    np.testing.assert_allclose(np.exp(individual).sum(-1) <= 1.00001, True)


def test_seed_count_is_explicit():
    assert seed_list() == [0]
    assert seed_list(3) == [0, 100, 200]
    for invalid in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            seed_list(invalid)


def test_natural_recipe_runs_only_requested_seeds(tmp_path, monkeypatch):
    from simpleb2t import experiments
    calls = []
    monkeypatch.setattr(experiments, "natural_evaluation", lambda work, model, seed, device: calls.append((model, seed)))
    monkeypatch.setattr(experiments, "aggregate", lambda work: None)
    experiments.run(tmp_path, "natural", device="cpu", seeds=[100])
    assert len(calls) == 5
    assert {seed for _, seed in calls} == {100}


def test_ablation_coverage_and_lm_seed_count(tmp_path, monkeypatch):
    from simpleb2t import evaluation
    from simpleb2t.workflows import ablations
    plan = ablation_plan()
    assert {r['beam'] for r in plan} == {1, 25, 50, 100, 200, 500}
    assert {r['evidence'] for r in plan} == {'embedding', 'individual', 'both'}
    calls = []
    monkeypatch.setattr(evaluation, 'decode', lambda *a, **k: calls.append((a, k)))
    ablations(tmp_path, seeds=[0, 100, 200], device='cpu', plan=plan[:3])
    assert len(calls) == 7  # deterministic LM once; two neural conditions, three seeds
    assert all(kwargs['partition'] == 'Core' for _, kwargs in calls)


def test_core_filter_never_labels_partial_result_full(tmp_path, monkeypatch):
    from simpleb2t import evaluation
    words = [f'w{i}' for i in range(92)]
    groups = [dict(word='w0', sentence_id=1, word_position=1),
              dict(word='w1', sentence_id=101, word_position=1)]
    monkeypatch.setattr(evaluation, 'groups', lambda: {'test': groups})
    monkeypatch.setattr(evaluation, 'bundled', lambda name: {'clinical': words} if name == 'vocabularies.json' else {'temperature': .04})
    monkeypatch.setattr(evaluation, 'targets', lambda words: torch.eye(92))
    vectors = np.zeros((2, 5, 92), dtype='float32')
    vectors[0, :, 0] = 1
    vectors[1, :, 1] = 1
    source = tmp_path / 'vectors.npy'
    np.save(source, vectors)
    monkeypatch.setattr(evaluation, 'export', lambda *a, **kw: source)
    core = evaluation.decode(tmp_path, 'ours', 0, weight=0, partition='Core', device='cpu')
    expanded = evaluation.decode(tmp_path, 'ours', 0, weight=0, partition='Expanded', device='cpu')
    assert set(core['metrics']) == {'Core'}
    assert set(expanded['metrics']) == {'Expanded'}
    assert core['metrics']['Core']['word_error_rate'] == 0
    assert expanded['metrics']['Expanded']['word_error_rate'] == 0
