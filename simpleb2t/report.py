"""Summaries of computed measurements. No plotting or typesetting dependencies."""

import collections
import csv
from pathlib import Path
import numpy as np
from .io import bundled, read, write


def mean_sd(values):
    values = list(values)
    return dict(
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        n=len(values),
    )


def metadata(work):
    """Counts underlying the session, sentence, and hyperparameter tables."""
    records = bundled("recordings.json.gz")
    natural = bundled("natural.json.gz")
    groups = bundled("groups.json.gz")
    splits = {}
    for split, data in natural.items():
        ids = sorted({i["record"] for i in data["items"]})
        splits[split] = dict(
            recordings=len(ids),
            hours=sum(records[i]["duration"] for i in ids) / 3600,
            eligible_occurrences=len(data["items"]),
            sentences=len(data["sequences"]),
            sessions=sorted({records[i]["session"] for i in ids}, key=int),
        )
    stitched = bundled("stitched_train.json.gz")
    retained = sum(map(len, stitched["fallback"]))
    result = dict(
        splits=splits,
        clinical=dict(
            Core_sentences=100,
            Expanded_sentences=100,
            Core_positions=595,
            Expanded_positions=680,
            occurrences_per_position=5,
            unique_test_occurrences=6375,
        ),
        development=dict(sentences=50, positions=307),
        stitched_training=dict(
            retained_words=retained,
            original_words=len(natural["train"]["items"]),
            retained_percent=100 * retained / len(natural["train"]["items"]),
        ),
        hyperparameters=bundled("experiment.json"),
    )
    write(Path(work) / "results" / "metadata.json", result)
    return result


def summarize_natural(work):
    root = Path(work) / "results"
    buckets = collections.defaultdict(list)
    subjects = collections.defaultdict(list)
    for path in sorted(root.glob("*/*/*.json")):
        data = read(path)
        if "condition" not in data or "top1_balanced_accuracy" not in data:
            continue
        buckets[(data["condition"], data["model"])].append(data)
        if data["condition"].startswith("subject_"):
            subjects[(data["model"], data["seed"])].append(data["top1_balanced_accuracy"] * 100)
    rows = []
    for (condition, model), values in sorted(buckets.items()):
        row = dict(
            condition=condition,
            model=model,
            seeds=[r["seed"] for r in values],
            balanced_accuracy_percent=mean_sd(100 * r["top1_balanced_accuracy"] for r in values),
            observed_classes=sorted({r["observed_word_types"] for r in values}),
            examples=sorted({r["examples"] for r in values}),
        )
        rows.append(row)
    write(root / "natural_summary.json", rows)
    cross = []
    for model in sorted({m for m, _ in subjects}):
        perseed = [
            dict(seed=s, subjects=len(v), mean_balanced_accuracy_percent=float(np.mean(v)))
            for (m, s), v in sorted(subjects.items())
            if m == model
        ]
        cross.append(
            dict(
                model=model,
                per_seed=perseed,
                equal_subject_mean=mean_sd(r["mean_balanced_accuracy_percent"] for r in perseed),
            )
        )
    write(root / "cross_subject_summary.json", cross)
    if rows:
        with (root / "natural_summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "condition",
                    "model",
                    "seeds",
                    "balanced_accuracy_mean_percent",
                    "sample_sd_percent",
                    "observed_classes",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["condition"],
                        r["model"],
                        r["seeds"],
                        r["balanced_accuracy_percent"]["mean"],
                        r["balanced_accuracy_percent"]["sd"],
                        r["observed_classes"],
                    ]
                )
    # The units of these analyses remain proportions (accuracy) and bits (JS).
    summaries = {}
    path = Path(work) / "analysis" / "timing_agreement.json"
    if path.exists():
        rows = read(path)
        summaries["timing_agreement"] = {
            kind: mean_sd(r["delta_js_bits"] for r in rows if r["model"] == kind)
            for kind in sorted({r["model"] for r in rows})
        }
    path = Path(work) / "analysis" / "future_context.json"
    if path.exists():
        buckets = collections.defaultdict(list)
        for r in read(path)["runs"]:
            buckets[(r["condition"], r["split"], r["half"])].append(r["word_accuracy"])
        summaries["future_context"] = [
            dict(condition=c, split=s, stratum=h, accuracy=mean_sd(v))
            for (c, s, h), v in buckets.items()
        ]
    write(root / "analysis_summary.json", summaries)
