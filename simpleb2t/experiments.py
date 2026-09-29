"""The paper's experiment list. Every recipe writes raw numerical results only."""

from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .io import bundled, read, write
from .evaluation import decode, export, targets, natural_evaluation


def aggregate(work):
    """Machine-readable summaries; replicas pool within seed, never across seeds."""
    import csv

    work = Path(work)
    rows = []
    for p in sorted((work / "decoding").glob("*/result.json")):
        r = read(p)
        for split, m in r["metrics"].items():
            rows.append(
                dict(
                    r["settings"],
                    experiment=p.parent.name,
                    partition=split,
                    WER=100 * m["word_error_rate"],
                    SMR=100 * m["sentence_exact_accuracy"],
                    BAcc=100 * m["top1_balanced_accuracy"],
                    word_accuracy=100 * m["top1_word_accuracy"],
                    evaluated_sentences=len(read(p.parent / "predictions.json")),
                )
            )
    write(work / "results" / "decoding.json", rows)
    if rows:
        with (work / "results" / "decoding.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # Group by actual model/settings/partition, retaining seed counts explicitly.
    buckets = {}
    for r in rows:
        key = tuple(
            (k, str(v))
            for k, v in r.items()
            if k
            not in {
                "experiment",
                "seed",
                "WER",
                "SMR",
                "BAcc",
                "word_accuracy",
                "evaluated_sentences",
            }
        )
        buckets.setdefault(key, []).append(r)
    summary = []
    for key, rs in buckets.items():
        # An experiment can also be reused by an ablation; do not count a seed twice.
        byseed = {r["seed"]: r for r in rs}
        rs = list(byseed.values())
        row = dict(key)
        row["seeds"] = list(byseed)
        for metric in ["WER", "SMR", "BAcc", "word_accuracy"]:
            vals = [r[metric] for r in rs]
            row[metric] = dict(
                mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        summary.append(row)
    write(work / "results" / "summary.json", summary)
    from .report import summarize_natural

    summarize_natural(work)
    return summary


def tune_baseline(work, kind, device, seeds=(0,)):
    """Separate calibration on validation and LM weighting on clinical development."""
    work = Path(work)
    path = work / "settings" / f"{kind}.json"
    if path.exists():
        result = read(path)
        if result["seeds"] != list(seeds):
            raise ValueError("Calibration seed list changed; use a fresh work directory.")
        return result
    temps = np.geomspace(0.0001, 10000, 401)
    curves = {1: [], 5: []}
    words = bundled("vocabularies.json")["clinical"]
    tgt = targets(words)
    import torch
    from torch.nn import functional as F

    for seed in seeds:
        source = export(work, kind, seed, "calibration", device)
        vectors = torch.from_numpy(np.load(source))
        truth = read(source.with_suffix(".json"))["words"]
        labels = np.array([words.index(w) for w in truth])
        values = {
            1: (vectors @ tgt.T).numpy().reshape(-1, 92),
            5: (F.normalize(vectors.mean(1), dim=-1) @ tgt.T).numpy(),
        }
        for k, scores in values.items():
            y = np.repeat(labels, 5) if k == 1 else labels
            curves[k].append(
                [
                    float(
                        (logsumexp(scores / t, axis=-1) - (scores / t)[np.arange(len(y)), y]).mean()
                    )
                    for t in temps
                ]
            )
    selected = {}
    weights = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0, 128.0, None]
    calibration = {str(k): float(temps[np.mean(curves[k], axis=0).argmin()]) for k in [1, 5]}
    write(
        work / "settings" / f"{kind}_temperature.json",
        dict(
            temperatures=calibration,
            grid=temps.tolist(),
            nll=curves,
            split="validation excluding development target donors",
        ),
    )
    for k in [1, 5]:
        trials = []
        t = calibration[str(k)]
        for weight in weights:
            edits = 0
            for seed in seeds:
                r = decode(
                    work,
                    kind,
                    seed,
                    k,
                    1.0 if weight is None else weight,
                    t,
                    0.0,
                    split="dev",
                    control="lm" if weight is None else "brain",
                    device=device,
                    tag=f"tune_{kind}_{seed}_{k}_{weight}",
                )
                # LM-only is deterministic, but match the five replicas in k=1 pooling.
                edits += r["metrics"]["Full"]["word_edits"] * (
                    5 if k == 1 and weight is None else 1
                )
            trials.append(dict(weight=weight, edits=edits))
        best = min(
            trials,
            key=lambda r: (r["edits"], -(float("inf") if r["weight"] is None else r["weight"])),
        )
        selected[str(k)] = dict(temperature=t, weight=best["weight"], trials=trials)
    result = dict(model=kind, seeds=seeds, selected=selected, test_used=False)
    write(path, result)
    return result


def tune_prompt(work, prompt, device):
    path = Path(work) / "settings" / f"prompt_{prompt}.json"
    if path.exists():
        return read(path)["weight"]
    if prompt == "E":
        return 0.5
    trials = []
    for weight in [0.0, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0]:
        r = decode(
            work,
            "ours",
            0,
            5,
            weight,
            prompt=prompt,
            split="dev",
            device=device,
            tag=f"tune_prompt_{prompt}_{weight}",
        )
        trials.append(dict(weight=weight, edits=r["metrics"]["Full"]["word_edits"]))
    selected = min(trials, key=lambda r: (r["edits"], r["weight"]))
    write(path, dict(weight=selected["weight"], trials=trials, split="development", seed=0))
    return selected["weight"]


def run(work, recipe, device="cuda", published_settings=False, seeds=(0,)):
    """Sequential execution fits a single local GPU; completed stages are reused."""
    from .training import run as train
    from .analysis import clinical_diagnostics, word_diagnostics, future_context, timing_agreement

    cfg = bundled("experiment.json")
    work = Path(work)
    if recipe == "metadata":
        from .report import metadata

        metadata(work)
    elif recipe == "train":
        for kind, spec in cfg["runs"].items():
            for seed in seeds:
                train(work, kind, seed, device)
    elif recipe == "natural":
        for kind in ["joint", "single_word", "shared_pulses", "independent_pulses", "timing"]:
            for seed in seeds:
                natural_evaluation(work, kind, seed, device)
    elif recipe == "clinical":
        decode(work, "ours", 0, 5, control="lm", device=device, tag="lm")
        for kind in ["ours", "joint", "stitched"]:
            settings = (
                None
                if kind == "ours"
                else bundled("baseline_settings.json")[kind]
                if published_settings
                else tune_baseline(work, kind, device, seeds)
            )
            for k in [1, 5]:
                selected = (
                    dict(temperature=cfg["temperature"], weight=cfg["weights"][str(k)])
                    if settings is None
                    else settings["selected"][str(k)]
                )
                for seed in seeds:
                    decode(
                        work,
                        kind,
                        seed,
                        k,
                        0.0,
                        selected["temperature"],
                        0.0,
                        device=device,
                        tag=f"brain_{kind}_{seed}_k{k}",
                    )
                    w = selected["weight"]
                    decode(
                        work,
                        kind,
                        seed,
                        k,
                        1.0 if w is None else w,
                        selected["temperature"],
                        2.0 if kind == "ours" else 0.0,
                        control="lm" if w is None else "brain",
                        device=device,
                        tag=f"clinical_{kind}_{seed}_k{k}",
                    )
    elif recipe == "ablations":
        from .workflows import clinical, ablations
        clinical(work, seeds=seeds, device=device)
        ablations(work, seeds=seeds, device=device)
    elif recipe == "observations":
        for k in range(1, 6):
            for seed in seeds:
                decode(
                    work,
                    "ours",
                    seed,
                    k,
                    weight=cfg["weights"][str(k)],
                    device=device,
                    tag=f"curve_k{k}_{seed}",
                )
        # Development curves for Figure 7, using seed 0 and all k-subsets.
        for k in range(1, 6):
            for weight in [0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0]:
                decode(
                    work,
                    "ours",
                    0,
                    k,
                    weight,
                    split="dev",
                    device=device,
                    tag=f"weight_k{k}_{weight}",
                )
    elif recipe == "prompts":
        for prompt in ["A", "B", "C", "D", "E", "F"]:
            weight = (
                bundled("prompt_settings.json")[prompt]
                if published_settings
                else tune_prompt(work, prompt, device)
            )
            decode(
                work,
                "ours",
                0,
                control="lm",
                prompt=prompt,
                device=device,
                tag="lm" if prompt == "E" else f"prompt_{prompt}_lm",
            )
            for seed in seeds:
                decode(
                    work,
                    "ours",
                    seed,
                    weight=weight,
                    prompt=prompt,
                    device=device,
                    tag=f"clinical_ours_{seed}_k5" if prompt == "E" else f"prompt_{prompt}_{seed}",
                )
    elif recipe == "cross-subject":
        for subject in range(1, 33):
            for kind in ["joint", "shared_pulses"]:
                for seed in seeds:
                    natural_evaluation(work, kind, seed, device, subject=subject)
    elif recipe == "stitched":
        for kind in ["joint", "stitched", "ours"]:
            for seed in seeds:
                for stitched in [False, True]:
                    natural_evaluation(
                        work, kind, seed, device, stitched=stitched, matched=not stitched
                    )
    elif recipe == "timing-agreement":
        timing_agreement(work, device, seeds)
    elif recipe == "diagnostics":
        clinical_diagnostics(work, seeds)
        word_diagnostics(work, device, seeds)
        future_context(work, device, seeds)
    elif recipe == "all":
        for name in [
            "metadata",
            "train",
            "natural",
            "clinical",
            "ablations",
            "observations",
            "prompts",
            "cross-subject",
            "stitched",
            "timing-agreement",
            "diagnostics",
        ]:
            run(work, name, device, published_settings, seeds)
    else:
        raise ValueError(recipe)
    aggregate(work)
