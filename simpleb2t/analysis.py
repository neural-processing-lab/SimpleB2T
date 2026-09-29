"""Numeric analyses for Figures 3, 5, 8 and Table 4; never renders figures."""

import collections
import re
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp, xlogy
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
from .io import DATA, bundled, read, write
from .data import Windows, natural, groups, records
from .training import load_model, predict
from .evaluation import targets
from .metrics import word_edit_distance


def js(p, q):
    m = (p + q) / 2
    return (0.5 * (xlogy(p, p) - xlogy(p, m) + xlogy(q, q) - xlogy(q, m))).sum(-1) / np.log(2)


def timing_agreement(work, device="cuda", seeds=(0,)):
    """Fit duration posterior on train, neural temperature on val; compare on test."""
    work = Path(work)
    words = bundled("vocabularies.json")["natural"]
    lookup = {w: i for i, w in enumerate(words)}
    a = Windows(work, "train", "timing")
    intervals = a[list(range(len(a)))].astype("float64")
    # Retain published double precision for fitting duration distributions.
    intervals = np.asarray(
        [a.annotations[i["record"]][str(i["event"])] for i in a.items], dtype="float64"
    )
    logs = [
        np.log(
            intervals[
                [j for j, i in enumerate(a.items) if i["word"] == w and np.isfinite(intervals[j])]
            ]
        )
        for w in words
    ]
    mean = np.array([v.mean() for v in logs])
    sd = np.array([v.std() for v in logs])
    prior = np.array([len(v) for v in logs])
    prior = prior / prior.sum()
    rng = np.random.default_rng(20260924)
    splits = {}
    for split in ["val", "test"]:
        arr = Windows(work, split, "timing")
        raw = np.asarray(
            [arr.annotations[i["record"]][str(i["event"])] for i in arr.items], dtype="float64"
        )
        ids = np.array(
            [j for j, i in enumerate(arr.items) if i["word"] in lookup and np.isfinite(raw[j])]
        )
        labels = np.array([lookup[arr.items[j]["word"]] for j in ids])
        ll = -0.5 * ((np.log(raw[ids, None]) - mean) / sd) ** 2 - np.log(sd) + np.log(prior)
        q = np.exp(ll - logsumexp(ll, axis=1, keepdims=True)).astype("float32")
        order = rng.permutation(len(ids))
        for label in range(50):
            rng.permutation(
                np.flatnonzero(labels == label)
            )  # same RNG stream as original within-word check
        splits[split] = (ids, labels, q, order)
    result = []
    for kind in ["joint", "shared_pulses", "ours"]:
        for seed in seeds:
            path = work / "analysis" / "timing_agreement" / f"{kind}_{seed}.json"
            if path.exists():
                result.append(read(path))
                continue
            torch.set_float32_matmul_precision("high")
            model, pos, _ = load_model(work, kind, seed, device)
            scores = {}
            for split in ["val", "test"]:
                arr = Windows(work, split, kind)
                v = predict(model, kind, arr, arr.sequences, device, pos)
                scores[split] = (v[splits[split][0]] @ targets(words).T).numpy().astype("float64")
            labels = splits["val"][1]

            def nll(logt):
                s = scores["val"] / np.exp(logt)
                return float((logsumexp(s, axis=1) - s[np.arange(len(s)), labels]).mean())

            t = float(np.exp(minimize_scalar(nll, bounds=(-9, 3), method="bounded").x))
            s = scores["test"] / t
            p = np.exp(s - logsumexp(s, axis=1, keepdims=True))
            q = splits["test"][2]
            actual = float(js(p, q).mean())
            shuffled = float(js(p, q[splits["test"][3]]).mean())
            row = dict(
                model=kind,
                seed=seed,
                temperature=t,
                actual_js_bits=actual,
                shuffled_js_bits=shuffled,
                delta_js_bits=shuffled - actual,
                n_test=len(q),
            )
            write(path, row)
            result.append(row)
            del model
    write(work / "analysis" / "timing_agreement.json", result)
    return result


def clinical_predictions(work, condition="combined", seed=0):
    tag = (
        "lm"
        if condition == "lm"
        else f"clinical_ours_{seed}_k5"
        if condition == "combined"
        else f"brain_ours_{seed}_k5"
    )
    rows = read(Path(work) / "decoding" / tag / "predictions.json")
    return sorted(
        [r for r in rows if r["original_sentence_id"] <= 100],
        key=lambda r: r["original_sentence_id"],
    )


def clinical_diagnostics(work, seeds=(0,)):
    """Per-seed oracle WER, positional confusion, POS accuracy, sentence WER."""
    words = bundled("vocabularies.json")["clinical"]
    lookup = {w: i for i, w in enumerate(words)}
    tags = [t for t in bundled("pos_tags.json")["tags"] if t["sentence_id"] <= 100]
    result = []
    for seed in seeds:
        for condition in ["brain", "lm", "combined"]:
            if condition == "lm" and seed != seeds[0]:
                continue
            rows = clinical_predictions(work, condition, seed)
            ref = [w for r in rows for w in r["reference"].split()]
            hyp = [w for r in rows for w in r["prediction"].split()]
            assert ref == [t["word"] for t in tags]
            matrix = np.zeros((92, 92), dtype=int)
            for a, b in zip(ref, hyp):
                matrix[lookup[a], lookup[b]] += 1
            categories = {}
            for cat in ["Nouns", "Verbs", "Adjectives", "Adverbs", "Function"]:
                ids = [i for i, t in enumerate(tags) if t["category"] == cat]
                categories[cat] = dict(
                    n=len(ids), accuracy=float(np.mean([ref[i] == hyp[i] for i in ids]))
                )
            row = dict(
                seed=seed,
                condition=condition,
                confusion_counts=matrix.tolist(),
                pos=categories,
                sentence_wer=[
                    word_edit_distance(r["reference"].split(), r["prediction"].split())
                    / len(r["reference"].split())
                    for r in rows
                ],
            )
            if condition == "combined":
                row["oracle_wer"] = {
                    str(n): sum(
                        min(
                            word_edit_distance(r["reference"].split(), c["prediction"].split())
                            for c in r["candidates"][:n]
                        )
                        for r in rows
                    )
                    / len(ref)
                    for n in [1, 5, 10, 25, 50]
                }
            result.append(row)
    from .report import mean_sd

    combined = [r for r in result if r["condition"] == "combined"]
    pooled = np.sum([r["confusion_counts"] for r in combined], axis=0)
    normalized = pooled / pooled.sum(axis=1, keepdims=True)
    histogram, edges = np.histogram(combined[0]["sentence_wer"], bins=np.linspace(0, 1, 11))
    summary = dict(
        oracle_wer={
            str(n): mean_sd(r["oracle_wer"][str(n)] for r in combined) for n in [1, 5, 10, 25, 50]
        },
        confusion_row_proportions=normalized.tolist(),
        histogram_seed=seeds[0],
        histogram_counts=histogram.tolist(),
        histogram_bin_edges=edges.tolist(),
        pos={
            condition: {
                category: mean_sd(
                    r["pos"][category]["accuracy"] for r in result if r["condition"] == condition
                )
                for category in ["Nouns", "Verbs", "Adjectives", "Adverbs", "Function"]
            }
            for condition in ["brain", "lm", "combined"]
        },
    )
    # Fixed illustrative targets from the paper; predictions always come from this run.
    examples = []
    by_reference = {r["reference"]: r for r in clinical_predictions(work, "combined", 0)}
    for label, reference in [
        ("Best", "i would like to be on my own"),
        ("Upper quartile", "can you give me more time"),
        ("Median", "i don't see your face"),
        ("Lower quartile", "do not put that on my back"),
    ]:
        row = by_reference[reference]
        examples.append(
            dict(
                label=label,
                reference=reference,
                prediction=row["prediction"],
                wer=word_edit_distance(reference.split(), row["prediction"].split())
                / len(reference.split()),
            )
        )
    write(
        Path(work) / "analysis" / "clinical.json",
        dict(words=words, runs=result, summary=summary, examples=examples),
    )
    return result


def word_diagnostics(work, device="cuda", seeds=(0,)):
    from .lm import QwenScorer
    from .evaluation import targets

    work = Path(work)
    words = bundled("vocabularies.json")["clinical"]
    lookup = {w: i for i, w in enumerate(words)}
    counts = collections.Counter()
    durations = collections.defaultdict(list)
    train = natural()["train"]["items"]
    for group in groups()["train"]:
        for i in group["indices"]:
            w = train[i]["word"]
            counts[w] += 1
            durations[w].append(train[i]["duration"])
    accuracies = []
    for seed in seeds:
        correct = collections.Counter()
        total = collections.Counter()
        for row in clinical_predictions(work, "combined", seed):
            for a, b in zip(row["reference"].split(), row["prediction"].split()):
                total[a] += 1
                correct[a] += a == b
        accuracies.append([correct[w] / total[w] for w in words])
    accuracy = np.mean(accuracies, axis=0)
    phones = collections.defaultdict(set)
    for line in (DATA / "pronunciations.dict").read_text().splitlines():
        p = line.split("#")[0].split()
        if p:
            phones[re.sub(r"\(\d+\)$", "", p[0])].add(tuple(re.sub(r"\d", "", x) for x in p[1:]))
    phonetic = [
        min(
            word_edit_distance(a, b) / max(len(a), len(b))
            for v in words
            if v != w
            for a in phones[w]
            for b in phones[v]
        )
        for w in words
    ]
    emb = targets(words).numpy()
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    semantic = 1 - sim.max(1)
    path = work / "analysis" / "lm_predictability.json"
    if path.exists():
        predictability = read(path)
    else:
        lm = QwenScorer(
            "Qwen/Qwen3-8B-Base",
            words,
            work / "lm_cache" / "E.pkl",
            device=device,
            revision=bundled("experiment.json")["qwen_revision"],
        )
        probs = collections.defaultdict(list)
        for row in clinical_predictions(work, "combined", 0):
            prefix = ()
            for w in row["reference"].split():
                i = lookup[w]
                probs[w].append(float(np.exp(lm.score_words([prefix])[0, i])))
                prefix += (i,)
        predictability = {w: float(np.mean(probs[w])) for w in words}
        write(path, predictability)
        lm.save()
    features = dict(
        phonetic_distinguishability=phonetic,
        semantic_distinguishability=semantic.tolist(),
        mean_duration_ms=[1000 * np.mean(durations[w]) for w in words],
        duration_sd_ms=[1000 * np.std(durations[w], ddof=1) for w in words],
        lm_predictability=[predictability[w] for w in words],
        log10_training_frequency=[float(np.log10(counts[w])) for w in words],
    )
    correlations = {
        k: dict(
            r=float(pearsonr(v, accuracy)[0]),
            slope=float(np.polyfit(v, accuracy, 1)[0]),
            intercept=float(np.polyfit(v, accuracy, 1)[1]),
        )
        for k, v in features.items()
    }
    bins = {
        label: [w for w, a in zip(words, accuracy) if lo <= a < hi or hi == 1 and a == 1]
        for label, lo, hi in [
            ("0-25%", 0, 0.25),
            ("25-50%", 0.25, 0.5),
            ("50-75%", 0.5, 0.75),
            ("75-100%", 0.75, 1),
        ]
    }
    result = dict(
        words=words,
        accuracy=accuracy.tolist(),
        accuracy_seed_sd=np.std(accuracies, axis=0, ddof=1).tolist(),
        training_counts=[counts[w] for w in words],
        features=features,
        correlations=correlations,
        accuracy_bins=bins,
    )
    write(work / "analysis" / "words.json", result)
    return result


def future_context(work, device="cuda", seeds=(0,)):
    """Likelihood of observed future transcript given each candidate; diagnostic only."""
    from .lm import QwenScorer

    work = Path(work)
    words = bundled("vocabularies.json")["clinical"]
    cfg = bundled("experiment.json")
    lm = QwenScorer(
        "Qwen/Qwen3-8B-Base",
        words,
        work / "lm_cache" / "E.pkl",
        device=device,
        revision=cfg["qwen_revision"],
    )
    base_text = ["Text: " + ("I" if w == "i" else w) for w in words]
    bases = [tuple(lm.tokenizer.encode(t, add_special_tokens=False)) for t in base_text]
    path = work / "analysis" / "future_occurrences.json"
    saved = read(path) if path.exists() else {}
    test = natural()["test"]["items"]
    recs = records()
    for g in groups()["test"][:595]:
        for member, idx in enumerate(g["indices"]):
            key = f"{g['sentence_id']}/{g['word_position']}/{member}"
            if key in saved:
                continue
            item = test[idx]
            r = recs[item["record"]]
            start = r["origin"] + round(item["onset"] * 50) / 50
            future = [
                e["text"]
                for e in sorted(r["words"], key=lambda e: (e["start"], e["event_index"]))
                if e["start"] > r["origin"] + item["onset"]
                and e["start"] + e["duration"] <= start + 3
            ]
            text = "".join(" " + ("I" if w == "i" else w) for w in future)
            pieces = tuple(lm.tokenizer.encode(text, add_special_tokens=False))
            if pieces:
                for b, t in zip(bases, base_text):
                    assert (
                        tuple(lm.tokenizer.encode(t + text, add_special_tokens=False)) == b + pieces
                    )
                ll = lm.continuation_likelihood(bases, pieces)
            else:
                ll = np.zeros(92)
            q = np.exp(ll - logsumexp(ll))
            saved[key] = dict(
                future=future,
                posterior=q.tolist(),
                target_probability=float(q[words.index(g["word"])]),
            )
            write(path, saved)
    positions = groups()["test"][:595]
    p = np.array(
        [
            np.mean(
                [
                    saved[f"{g['sentence_id']}/{g['word_position']}/{m}"]["target_probability"]
                    for m in range(5)
                ]
            )
            for g in positions
        ]
    )
    ordered = sorted(
        range(len(p)),
        key=lambda i: (p[i], positions[i]["sentence_id"], positions[i]["word_position"]),
    )
    halves = {
        "overall": {"low": ordered[:297], "high": ordered[297:]},
        "word_matched": {"low": [], "high": []},
    }
    for w in words:
        ids = [i for i in ordered if positions[i]["word"] == w]
        n = len(ids) // 2
        if n:
            halves["word_matched"]["low"] += ids[:n]
            halves["word_matched"]["high"] += ids[-n:]
    rows = []
    for condition in ["brain", "lm", "combined"]:
        for seed in [0] if condition == "lm" else seeds:
            pred = clinical_predictions(work, condition, seed)
            correct = np.array(
                [
                    a == b
                    for row in pred
                    for a, b in zip(row["reference"].split(), row["prediction"].split())
                ]
            )
            for split, parts in halves.items():
                for half, ids in parts.items():
                    rows.append(
                        dict(
                            condition=condition,
                            seed=seed,
                            split=split,
                            half=half,
                            n=len(ids),
                            word_accuracy=float(correct[ids].mean()),
                        )
                    )
    result = dict(
        quantiles=np.quantile(p, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        positions=p.tolist(),
        strata=halves,
        runs=rows,
    )
    write(work / "analysis" / "future_context.json", result)
    return result
