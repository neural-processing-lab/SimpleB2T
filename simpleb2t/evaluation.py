"""Frozen neural exports and sentence decoding, with explicit observation subsets."""

import collections
import itertools
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .io import DATA, bundled, read, write, checkpoint, sha
from .data import Windows, ClinicalWindows, groups, cross_subject_metadata
from .training import load_model, forward, predict, validation_vectors
from .metrics import metrics


def targets(vocabulary):
    archive = np.load(DATA / "targets.npz")
    lookup = {w: i for i, w in enumerate(archive["words"].tolist())}
    return F.normalize(
        torch.from_numpy(archive["embeddings"][[lookup[w] for w in vocabulary]]), dim=-1
    )


def export(work, kind, seed, split="test", device="cuda", noise=False):
    """Save [word position, occurrence, embedding] before any LM operation."""
    work = Path(work)
    folder = work / "embeddings" / kind / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (split + ("_noise" if noise else "") + ".npy")
    if path.exists():
        return path
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    model, pos, saved = load_model(work, kind, seed, device)
    if split == "calibration":
        a = Windows(work, "val", kind)
        v, truth, donors = validation_vectors(model, kind, a, device, pos)
        forbidden = {i for row in bundled("stitched_dev.json")["donors"] for i in row}
        words = bundled("vocabularies.json")["clinical"]
        selected = [
            i for i, w in enumerate(truth) if w in words and not forbidden.intersection(donors[i])
        ]
        v = v[selected]
        write(
            path.with_suffix(".json"),
            dict(words=[truth[i] for i in selected], excluded=len(truth) - len(selected)),
        )
    else:
        g = groups()[split]
        repaired_dev = split == "dev" and kind in ("joint", "stitched")
        if kind in ("ours", "joint", "stitched", "single_word") and not repaired_dev:
            a = ClinicalWindows(work, split)
        else:
            a = Windows(work, "val" if split == "dev" else "test", kind)
        indices = [p["indices"] for p in g]
        if split == "dev" and kind in ("joint", "stitched"):
            indices = bundled("stitched_dev.json")["donors"]
        v = torch.empty(len(g), 5, 1024)
        if kind == "ours":
            moments = None
            if noise:
                moments = noise_moments(work)
                rng = np.random.default_rng(20260918)
            # Core/Expanded were exported separately; retain that batching boundary.
            boundaries = [(0, 595), (595, len(g))] if split == "test" else [(0, len(g))]
            with torch.inference_mode():
                for lo, hi in boundaries:
                    for start in range(lo, hi, 25):
                        selected = indices[start : min(start + 25, hi)]
                        n = len(selected)
                        if noise:
                            x = (
                                rng.standard_normal((125, 306, 150), dtype=np.float32)
                                * moments["std"]
                                + moments["mean"]
                            )
                        else:
                            selected = selected + [selected[0]] * (25 - n)
                            x = a[[i for group in selected for i in group]]
                        pred = forward(model, kind, torch.from_numpy(x).to(device), pos, [125])[
                            : n * 5
                        ]
                        v[start : start + n] = pred.reshape(n, 5, 1024).cpu()
        else:
            seqs = collections.defaultdict(list)
            for j, p in enumerate(g):
                for m in range(5):
                    seqs[(m, p["sentence_id"])].append(
                        (p["word_position"], indices[j][m], j * 5 + m)
                    )
            ordered = [
                [(source, dest) for _, source, dest in sorted(rows)]
                for _, rows in sorted(seqs.items())
            ]
            v = predict(model, kind, a, ordered, device, pos, True).reshape(len(g), 5, 1024)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, v.numpy())
    temporary.replace(path)
    write(
        path.with_suffix(".receipt.json"),
        dict(
            model=kind,
            seed=seed,
            split=split,
            noise=noise,
            selected_epoch=saved["epoch"],
            checkpoint_sha256=sha(checkpoint(work, kind, seed)),
        ),
    )
    return path


def noise_moments(work):
    path = Path(work) / "noise_moments.npz"
    if not path.exists():
        a = Windows(work, "train")
        order = [i for g in groups()["train"] for i in g["indices"]]
        indices = np.sort(np.random.default_rng(20260917).choice(len(order), 10000, replace=False))
        total = np.zeros((306, 150))
        squares = np.zeros_like(total)
        for start in range(0, len(indices), 100):
            x = a[[order[i] for i in indices[start : start + 100]]].astype("float64")
            total += x.sum(0)
            squares += (x * x).sum(0)
        mean = total / len(indices)
        std = np.sqrt(np.maximum(squares / len(indices) - mean**2, 0))
        np.savez(path, mean=mean.astype("float32"), std=std.astype("float32"), indices=indices)
    return np.load(path)


def natural_evaluation(
    work, kind, seed, device="cuda", subject=None, stitched=False, matched=False
):
    label = (
        f"subject_{subject}"
        if subject is not None
        else "stitched"
        if stitched
        else "matched_natural"
        if matched
        else "natural"
    )
    out = Path(work) / "results" / label / kind / f"{seed}.json"
    if out.exists():
        return read(out)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(8)
    model, pos, saved = load_model(work, kind, seed, device)
    metadata = cross_subject_metadata(work, subject) if subject is not None else None
    array = Windows(work, "test", kind, metadata)
    if stitched or matched:
        definition = bundled("stitched_test.json.gz")["sentences"]
        sequences = []
        truth = []
        offset = 0
        for row in definition:
            ids = row["donors"] if stitched else row["source_indices"]
            sequences.append([(i, offset + j) for j, i in enumerate(ids)])
            offset += len(ids)
            truth.extend(row["words"])
        v = predict(model, kind, array, sequences, device, pos, True)
    else:
        v = predict(model, kind, array, array.sequences, device, pos)
        truth = [i["word"] for i in array.items]
    words = bundled("vocabularies.json")["natural"]
    allowed = set(words)
    ids = [i for i, w in enumerate(truth) if w in allowed]
    guesses = (v[ids] @ targets(words).T).argmax(-1)
    result = metrics([truth[i] for i in ids], [words[j] for j in guesses])
    result.update(
        model=kind, seed=seed, selected_epoch=saved["epoch"], candidate_count=50, condition=label
    )
    write(out, result)
    return result


def word_scores(vectors, members, temperature, alpha, word_targets, evidence="both"):
    """Equivalent to log p(mean embedding) + alpha * mean(log p(individual))."""
    from .lm import log_probabilities

    v = torch.from_numpy(np.asarray(vectors[:, list(members)], dtype="float32"))
    mean = F.normalize(v.mean(1), dim=-1)
    avg = (mean @ word_targets.T).numpy()
    individual = (v @ word_targets.T).numpy()
    embedding = log_probabilities(avg, temperature)
    occurrence = log_probabilities(individual, temperature).mean(1)
    if evidence == "embedding":
        return embedding
    if evidence == "individual":
        return occurrence
    if evidence != "both":
        raise ValueError(f"Unknown evidence: {evidence}")
    return embedding + alpha * occurrence


def summarize_rows(rows):
    reference = []
    predicted = []
    positions = []
    for i, row in enumerate(rows):
        ref = row["reference"].split()
        hyp = row["prediction"].split()
        assert len(ref) == len(hyp)
        reference.extend(ref)
        predicted.extend(hyp)
        positions.extend(dict(sentence_id=i + 1, word_position=j + 1) for j in range(len(ref)))
    m = metrics(reference, predicted, positions)
    m.pop("sentence_predictions")
    return m


def decode(
    work,
    kind,
    seed,
    k=5,
    weight=0.5,
    temperature=None,
    alpha=2.0,
    beam=50,
    prompt="E",
    split="test",
    control="brain",
    device="cuda",
    tag=None,
    lm=None,
    evidence="both",
    partition="Full",
):
    """No reference prefix is used. All k-subsets are pooled within each seed."""
    from .lm import QwenScorer, nbest, PROMPTS

    if k not in range(1, 6) or beam < 1:
        raise ValueError("Use 1 <= k <= 5 and a positive beam width.")
    if partition not in {"Core", "Expanded", "Full"}:
        raise ValueError("partition must be Core, Expanded, or Full")
    if evidence not in {"both", "embedding", "individual"}:
        raise ValueError("evidence must be both, embedding, or individual")
    cfg = bundled("experiment.json")
    temperature = cfg["temperature"] if temperature is None else temperature
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    settings = dict(
        model=kind,
        seed=seed,
        k=k,
        weight=weight,
        temperature=temperature,
        alpha=alpha,
        beam=beam,
        prompt=prompt,
        split=split,
        control=control,
        evidence=evidence,
        partition=partition,
    )
    key = tag or f"{kind}_{seed}_k{k}_w{weight}_a{alpha}_b{beam}_{prompt}_{split}_{control}"
    if tag is None:
        key += f"_{evidence}_{partition}"
    folder = Path(work) / "decoding" / key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "settings.json").exists():
        assert read(folder / "settings.json") == settings, (
            "Existing decoding tag has different settings"
        )
    if (folder / "result.json").exists():
        return read(folder / "result.json")
    write(folder / "settings.json", settings)
    g = groups()[split]
    words = bundled("vocabularies.json")["clinical"]
    vectors = None
    if control != "lm":
        source = export(work, kind, seed, split, device, noise=control == "noise")
        vectors = np.load(source)
        if control == "shuffled":
            # Published ablation shuffles the 2,975 Core occurrences only.
            assert split == "test"
            vectors = vectors[:595].copy()
            order = np.random.default_rng(20260919).permutation(595 * 5)
            vectors = vectors.reshape(-1, 1024)[order].reshape(595, 5, 1024)
            g = g[:595]
    if (
        tag
        and (tag.startswith("ablation") or tag.startswith("curve") or tag.startswith("weight"))
        and split == "test"
    ):
        g = g[:595]
        if vectors is not None:
            vectors = vectors[:595]
    if split == "test" and partition != "Full":
        selected = [i for i, item in enumerate(g)
                    if (item["sentence_id"] <= 100) == (partition == "Core")]
        g = [g[i] for i in selected]
        if vectors is not None:
            vectors = vectors[selected]
    grouped = collections.defaultdict(list)
    for i, p in enumerate(g):
        grouped[p["sentence_id"]].append(i)
    for ids in grouped.values():
        ids.sort(key=lambda i: g[i]["word_position"])
    needs_lm = weight != 0 or control == "lm"
    if needs_lm and lm is None:
        lm = QwenScorer(
            "Qwen/Qwen3-8B-Base",
            words,
            Path(work) / "lm_cache" / f"{prompt}.pkl",
            prompt=PROMPTS[prompt],
            device=device,
            revision=cfg["qwen_revision"],
        )
    subsets = [tuple(range(5))] if control == "lm" else list(itertools.combinations(range(5), k))
    path = folder / "predictions.json"
    rows = read(path) if path.exists() else []
    done = {(r["replica"], r["original_sentence_id"]) for r in rows}
    word_targets = targets(words)
    for replica, members in enumerate(subsets):
        scores = (
            np.zeros((len(g), 92))
            if control == "lm"
            else word_scores(vectors, members, temperature, alpha, word_targets, evidence)
        )
        for sid, ids in sorted(grouped.items()):
            if (replica, sid) in done:
                continue
            if needs_lm:
                candidates = nbest(scores[ids], lm, 1.0 if control == "lm" else weight, beam)
            else:
                candidates = [(tuple(scores[ids].argmax(-1).tolist()), 0.0)]
            rows.append(
                dict(
                    replica=replica,
                    original_sentence_id=sid,
                    members=list(members),
                    reference=" ".join(g[i]["word"] for i in ids),
                    prediction=" ".join(words[i] for i in candidates[0][0]),
                    candidates=[
                        dict(prediction=" ".join(words[i] for i in seq), score=score)
                        for seq, score in candidates
                    ],
                )
            )
            write(path, rows)
            print(f"{key}: {len(rows)}/{len(grouped) * len(subsets)} sentences", flush=True)
        if lm is not None:
            lm.save()
    rows.sort(key=lambda r: (r["replica"], r["original_sentence_id"]))
    parts = {"Full": rows}
    if split == "test":
        parts.update(
            Core=[r for r in rows if r["original_sentence_id"] <= 100],
            Expanded=[r for r in rows if r["original_sentence_id"] > 100],
        )
    if split == "test" and not (parts.get("Core") and parts.get("Expanded")):
        parts.pop("Full", None)
    result = dict(
        settings=settings,
        metrics={s: summarize_rows(rr) for s, rr in parts.items() if rr},
        predictions=str(path.relative_to(Path(work))),
    )
    write(folder / "result.json", result)
    return result
