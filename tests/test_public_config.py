from simpleb2t import evaluation, experiments
from simpleb2t.io import bundled


def test_model_configurations_are_seed_independent():
    settings = bundled('experiment.json')
    assert len(settings['models']) == 7
    for config in settings['models'].values():
        assert 'seed' not in config
        assert config['max_epochs'] == 30
        assert config['patience'] == 10


def test_decoding_defaults_honor_common_and_specific_settings(monkeypatch):
    config = dict(temperature=.07, weights={'1':1.2, '5':.8}, alpha=3., beam=25,
                  baseline_decoding={'joint':{'5':{'temperature':.02, 'weight':4.}}},
                  prompt_weights={'A':.25})
    monkeypatch.setattr(evaluation, 'bundled', lambda name: config)
    assert evaluation.decoding_defaults('ours', 5) == dict(temperature=.07, weight=.8, alpha=3., beam=25)
    assert evaluation.decoding_defaults('ours', 1)['weight'] == 1.2
    assert evaluation.decoding_defaults('ours', 5, 'A')['weight'] == .25
    assert evaluation.decoding_defaults('joint', 5) == dict(temperature=.02, weight=4., alpha=0., beam=25)


def test_default_clinical_recipe_does_not_tune(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(experiments, 'decode', lambda *a, **kw: calls.append((a,kw)))
    monkeypatch.setattr(experiments, 'aggregate', lambda work: None)
    def unexpected(*args, **kwargs):
        raise AssertionError('Tuning must be explicitly requested')
    monkeypatch.setattr(experiments, 'tune_baseline', unexpected)
    experiments.run(tmp_path, 'clinical', device='cpu', seeds=[123])
    assert len(calls)==13
    assert all(a[2]==123 for a,kw in calls if kw.get('control')!='lm')


def test_tuning_is_explicit_and_receives_requested_seeds(tmp_path, monkeypatch):
    tuned=[]
    monkeypatch.setattr(experiments, 'decode', lambda *a, **kw: None)
    monkeypatch.setattr(experiments, 'aggregate', lambda work: None)
    def fit(work, kind, device, seeds):
        tuned.append((kind,seeds))
        return {'selected':{str(k):{'temperature':.1,'weight':1.} for k in (1,5)}}
    monkeypatch.setattr(experiments, 'tune_baseline', fit)
    experiments.run(tmp_path, 'clinical', device='cpu', seeds=[123], tune=True)
    assert tuned==[(model,[123]) for model in ['ours','joint','stitched']]
