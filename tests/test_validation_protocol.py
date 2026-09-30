import pytest
import torch
from simpleb2t import training
from simpleb2t.io import write


@pytest.mark.parametrize('kind', sorted(training.NATURAL_CONTROLS))
def test_controls_score_individual_occurrences_and_keep_sentence_context(monkeypatch, kind):
    class Words:
        items = [dict(word=w) for w in ['a','outside','a','b']]
        sequences = [[0,1,2,3]]
    seen = []
    def predict(model, mode, array, sequences, device, positions):
        seen.append(sequences)
        return torch.tensor([[1.,0.], [0.,1.], [0.,1.], [0.,1.]])
    monkeypatch.setattr(training, 'predict', predict)
    monkeypatch.setattr(training, 'validation_vectors', lambda *args: pytest.fail('Must not average'))
    score = training.validation_score(None, kind, Words(), 'cpu', None, torch.eye(2), ['a','b'])
    assert seen == [[[0,1,2,3]]]
    assert score == .75  # mean of recalls .5 and 1; not micro accuracy 2/3
    protocol = training.validation_protocol(kind)
    assert len(protocol['vocabulary']) == 50
    assert protocol['observations'] == 1


def test_simpleb2t_keeps_clinical_validation():
    protocol = training.validation_protocol('ours')
    assert len(protocol['vocabulary']) == 92
    assert protocol['observations'] == 5


def test_old_control_selection_cannot_silently_resume(tmp_path):
    folder = tmp_path / 'models/timing/0'
    write(folder / 'config.json', dict(model='timing'))
    write(folder / 'complete.json', dict(best_validation=.34))
    with pytest.raises(ValueError, match='Validation protocol changed'):
        training.run(tmp_path, 'timing', 0, device='cpu')
