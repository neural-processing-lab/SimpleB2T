"""Small notebook helpers. Model training and decoding stay in their own modules."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import DATA, bundled, read


def seed_list(count=1):
    """Use the paper's seed order, extending it for additional independent runs."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    return [100 * i for i in range(count)]


def storage_estimate():
    records = [r for r in bundled("recordings.json.gz") if r["subject"] == "0"]
    downloads = bundled("downloads.json")
    return pd.Series({
        "Raw recordings (GiB)": sum(downloads[r["neural"]]["size"] for r in records) / 2**30,
        "PNPL continuous cache (GiB)": sum(round((r["last_sample_time"] - r["origin"]) * 50) + 1 for r in records) * 306 * 8 / 2**30,
    }).round(1)


def overlap_summary():
    """Adjacent eligible pairs inside natural sentences, with no cross-sentence pairs."""
    from .data import natural
    rows = []
    for split, metadata in natural().items():
        gaps = np.array([metadata["items"][b]["onset"] - metadata["items"][a]["onset"]
                         for sentence in metadata["sequences"]
                         for a, b in zip(sentence, sentence[1:])])
        if np.any(gaps < 0):
            raise ValueError("Sentence onsets are not chronological")
        overlap = np.maximum(0.0, 3.0 - gaps)
        rows.append({"Split": split, "Adjacent pairs": len(gaps),
                     "Overlapping pairs (%)": 100 * np.mean(gaps < 3),
                     "Mean shared window (%)": 100 * np.mean(overlap / 3),
                     "Mean overlap (s)": overlap.mean()})
    return pd.DataFrame(rows).set_index("Split")


def overlap_example():
    """A toy identity check, not a fitted model or an empirical decoding result."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    rate, length, shift = 50, 150, 25
    signal = rng.normal(size=length + shift)
    first, second = signal[:length], signal[shift:]
    independent = rng.normal(size=length)
    np.testing.assert_array_equal(first[shift:], second[:-shift])
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
    for ax, following, title in zip(axes, [second, independent], ["Shared signal", "Independent signals"]):
        ax.plot(np.arange(length) / rate, first, lw=1, color="#2878B5", label="First window")
        ax.plot((np.arange(length) + shift) / rate, following + 5, lw=1, color="#E68632", label="Next window (offset)")
        ax.axvspan(shift / rate, length / rate, alpha=.10, color="grey")
        ax.set(xlabel="Time (s)", title=title, yticks=[])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("Signal + display offset")
    fig.tight_layout()
    return fig, {"Onset gap (s)": shift / rate,
                 "Shared fraction": (length - shift) / length,
                 "Shared samples agree": bool(np.array_equal(first[shift:], second[:-shift])),
                 "Independent samples agree": bool(np.array_equal(first[shift:], independent[:-shift]))}


def clinical(work, seeds=(0,), device="cuda", observations=(1, 5)):
    """Compute the three main conditions; all 200 sentences, published settings."""
    from .evaluation import decode
    settings = bundled("experiment.json")
    results = [decode(work, "ours", seeds[0], control="lm", device=device, tag="lm")]
    for k in observations:
        for seed in seeds:
            results.append(decode(work, "ours", seed, k=k, weight=0, alpha=0,
                                  device=device, tag=f"brain_ours_{seed}_k{k}"))
            results.append(decode(work, "ours", seed, k=k, weight=settings["weights"][str(k)],
                                  device=device, tag=f"clinical_ours_{seed}_k{k}"))
    return results


def ablation_plan():
    """Each row changes one setting at k=5. All ablation evaluations use Core."""
    rows = [dict(Experiment="LM", control="lm"),
            dict(Experiment="Brain", weight=0.0, alpha=0.0),
            dict(Experiment="Brain + LM"),
            dict(Experiment="Noise", control="noise"),
            dict(Experiment="Shuffled", control="shuffled")]
    rows += [dict(Experiment=f"Evidence: {value}", evidence=value)
             for value in ["embedding", "individual"]]
    rows += [dict(Experiment=f"Beam: {value}", beam=value) for value in [1, 25, 100, 200, 500]]
    rows += [dict(Experiment=f"LM weight: {value}", weight=value) for value in [0.0, .25, 1., 2.]]
    rows += [dict(Experiment=f"Individual weight: {value}", alpha=value) for value in [0., .5, 1., 4.]]
    defaults = dict(k=5, weight=.5, alpha=2., beam=50, evidence="both", control="brain", prompt="E")
    return [dict(defaults, **row) for row in rows]


def ablations(work, seeds=(0,), device="cuda", plan=None):
    from .evaluation import decode
    results = []
    for config in ablation_plan() if plan is None else plan:
        config = dict(config)
        label = config.pop("Experiment")
        for seed in seeds[:1] if config["control"] == "lm" else seeds:
            print(f"{label}; seed {seed}", flush=True)
            results.append(decode(work, "ours", seed, partition="Core", device=device, **config))
    return results


def decoding_table(work, seeds=(0,)):
    """One row per measured seed/partition; no substituted reference scores."""
    rows = []
    for path in sorted((Path(work) / "decoding").glob("*/result.json")):
        result = read(path)
        settings = result["settings"]
        if settings["seed"] not in seeds:
            continue
        for partition, value in result["metrics"].items():
            rows.append(dict(experiment=path.parent.name, **settings, subset=partition,
                             WER=100 * value["word_error_rate"],
                             SMR=100 * value["sentence_exact_accuracy"],
                             BAcc=100 * value["top1_balanced_accuracy"]))
    return pd.DataFrame(rows)


def natural_table(work, seeds=(0,)):
    rows = []
    for path in sorted((Path(work) / "results").glob("*/*/*.json")):
        r = read(path)
        if "condition" in r and r.get("seed") in seeds:
            rows.append({"Condition": r["condition"], "Model": r["model"], "Seed": r["seed"],
                         "Balanced accuracy (%)": 100 * r["top1_balanced_accuracy"],
                         "Observed classes": r["observed_word_types"]})
    return pd.DataFrame(rows)


def summarize(frame, by, value):
    """SD is undefined for one seed, rather than a zero-sized uncertainty estimate."""
    return frame.groupby(by, dropna=False)[value].agg(["mean", "std", "count"])
