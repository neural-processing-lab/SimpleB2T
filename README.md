# Removing Timing Shortcuts Improves Non-Invasive Brain-to-Text

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#citation)
[![Notebooks](https://img.shields.io/badge/Notebooks-get%20started-F37626.svg?logo=jupyter&logoColor=white)](#start-here)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

This repository contains the experiments for the paper "Removing Timing Shortcuts Improves Non-Invasive Brain-to-Text".

The code is organised around two notebooks. They explain the ideas, train the models, and let you try the main experiments on LibriBrain100. Everything runs locally, with one training seed by default.

```bibtex
@misc{simpleb2t,
  title  = {Removing Timing Shortcuts Improves Non-Invasive Brain-to-Text},
  author = {AUTHOR LIST},
  year   = {YEAR},
  note   = {Preprint. arXiv identifier forthcoming.}
}
```

## Start here

- **[Overlap and timing](notebooks/01_overlap_and_timing.ipynb)** shows how overlapping word windows expose timing, then tests joint decoding with real MEG, synthetic signals, and timing alone.
- **[SimpleB2T](notebooks/02_simpleb2t.ipynb)** trains an isolated word decoder, combines repeated observations, and reconstructs sentences with a frozen language model. It includes evaluation and ablations.

You can start with either notebook. They share downloaded data and checkpoints.

## Install

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m ipykernel install --user --name simpleb2t --display-name "Python (simpleb2t)"
jupyter lab notebooks/
```

Select the **Python (simpleb2t)** kernel.

## Run

Each notebook starts with a few settings:

```python
RUN = False       # Preview the examples without downloading recordings or training.
N_SEEDS = 1       # Increase for multiple training runs.
DEVICE = "cuda"
EXTRA = set()     # Enable optional experiments by name, as described in the notebook.
```

Set `RUN = True` to train and evaluate. Selected `EXTRA` experiments run independently of this flag, using the models named in their section. Set `BASE` to a directory with enough space for data and results.

The full workflow needs a CUDA GPU and roughly **200 GiB of free disk space**. We recommend 64 GiB of RAM and 48 GB of GPU memory for the FP32 Qwen decoder. The previews run on CPU. Synthetic and timing-only controls do not need raw MEG recordings.

The split, word targets, and benchmark assignments are included. Recordings and Qwen weights download when needed. Training saves checkpoints and resumes interrupted runs; predictions and scores are saved under `runs/`. Use a new `WORK` directory when changing an experiment.

## License

[CC BY-NC 4.0](LICENSE). The CNN, Transformer wrapper, and D-SigLIP loss derive from the implementation associated with d’Ascoli et al., *Towards decoding individual words from non-invasive brain recordings*. Dataset and model credits are in [third-party notices](THIRD_PARTY_NOTICES.md).

Data loading uses [PNPL](https://github.com/neural-processing-lab/pnpl): its `ClinicalCommunication` dataset supplies the benchmark and word windows. The optional other-subject analysis uses PNPL’s regular LibriBrain100 file access with the same preprocessing. Recordings are cached once; overlapping windows are extracted when needed.
