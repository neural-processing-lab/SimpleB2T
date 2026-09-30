# Finding your way around the code

Start with the notebooks. Their cells call the same Python functions you can use in your own scripts.

## The main path

1. **`data.py`** adapts PNPL loaders to batches of word windows.
2. **`models.py`** defines the isolated CNN + MLP and the joint Transformer decoder.
3. **`training.py`** trains a model and saves the best validation checkpoint.
4. **`evaluation.py`** predicts word embeddings, combines observations, and evaluates sentences.
5. **`lm.py`** scores candidate words with Qwen and runs beam search.

For example, after preparing the data:

```python
from simpleb2t.training import run as train
from simpleb2t.workflows import clinical

train("runs", "ours", seed=0, device="cuda")
clinical("runs", seeds=[0], device="cuda", observations=(1, 5))
```

`ours` is the isolated CNN + residual MLP. `joint` adds the sentence Transformer. `single_word` uses that Transformer on one word at a time.

## What to change

| To try… | Look in… |
|---|---|
| A different neural decoder | `models.py` |
| A different training objective or optimizer | `training.py` |
| Another way to combine repeated observations | `evaluation.py`, especially `word_scores` |
| A different prompt | `lm.py`, especially `PROMPTS` |
| Another beam width or LM weight | Arguments to `evaluation.decode` |
| A different synthetic signal | `synthetic.py` |
| Non-overlapping word assignments | `stitching.py` |

Use a fresh output directory when changing settings so that old cached results are not reused.

## Inputs and outputs

PNPL supplies the recording split, clinical donor assignments, and preprocessed windows.
`simpleb2t/assets/` holds frozen T5 targets and the metadata used by the optional analyses.
PNPL caches continuous recordings under `raw_data/.clinical_cache/`; it extracts windows on demand.

Under your chosen `WORK` directory:

- `models/` contains checkpoints and training histories.
- `decoding/` contains predicted sentences, beam candidates, and metrics.
- `results/` and `analysis/` contain summaries and diagnostic measurements.

`metrics.py` defines word error rate, sentence match, and balanced word accuracy. `analysis.py` contains the optional error analyses. `workflows.py` supplies the notebook helpers.

To check a code change, run `python -m pytest -q`. The tests cover the scoring rules, beam search, independent word processing, and training resumption.

## Configuration

`simpleb2t/assets/experiment.json` contains one training configuration per model under
`models`. Pass the seed to `train`; it does not need its own configuration.

Decoding defaults live in the same file: `temperature`, `weights` (LM weight by observation
count), `alpha`, `beam`, `baseline_decoding`, and `prompt_weights`. They are starting values
from our experiments, not automatically fitted to a new checkpoint. Arguments passed to
`decode` override them. Use a fresh output directory when changing them.

For optional tuning, run `python -m simpleb2t run clinical --tune` or
`python -m simpleb2t run prompts --tune`. Neural temperature uses validation data; LM
weight uses development sentences. Test data is not used for tuning.
