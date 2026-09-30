"""One local device, full-state resumption, validation-only checkpoint selection."""

import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .io import DATA, bundled, read, write, checkpoint
from .data import Windows, groups, bank
from .models import ClinicalCNN, SentenceCNN, ControlledModel, TimingModel, batches
from .vendor.simpleconv import SimpleConvTimeAggConfig
from .vendor.transformer import TransformerEncoderConfig
from .vendor.loss import SigLipLoss
from .metrics import metrics


def build(kind, settings, device, timing=None):
    cfg = bundled("experiment.json")
    if kind == "timing":
        model = TimingModel(TransformerEncoderConfig(**cfg["transformer"]).build(dim=1024), timing)
    else:
        cnn = SimpleConvTimeAggConfig(**cfg["cnn"]).build(n_in_channels=306, n_outputs=1024)
        if kind == "ours":
            model = ClinicalCNN(cnn)
        else:
            model = SentenceCNN(cnn, TransformerEncoderConfig(**cfg["transformer"]).build(dim=1024))
            if kind in ("single_word", "shared_pulses", "independent_pulses"):
                model = ControlledModel(model, kind)
    return model.to(device)


def forward(model, kind, x, positions, lengths):
    return (
        model(x, positions.expand(len(x), -1, -1))
        if kind == "ours"
        else model(x, positions, lengths)
    )


def predict(model, kind, array, sequences, device, positions, mapped=False):
    """Mapped entries are (source index, destination index), allowing repeated donors."""
    model.eval()
    n = max((v[1] for seq in sequences for v in seq), default=-1) + 1 if mapped else len(array)
    output = torch.empty(n, 1024)
    with torch.inference_mode():
        for packed in batches(sequences):
            rows = [v for seq in packed for v in seq]
            src = [r[0] for r in rows] if mapped else rows
            dst = [r[1] for r in rows] if mapped else rows
            x = torch.from_numpy(array[src]).to(device)
            output[dst] = forward(model, kind, x, positions, [len(s) for s in packed]).cpu()
    return output


def validation_vectors(model, kind, array, device, positions):
    if kind == "stitched":
        definition = bundled("stitched_val.json.gz")
        seqs = []
        truth = []
        donors = []
        offset = 0
        for sentence in definition["sentences"]:
            n = len(sentence["words"])
            truth.extend(sentence["words"])
            donors.extend([[sentence["versions"][m][j] for m in range(5)] for j in range(n)])
            for m in range(5):
                seqs.append(
                    [(v, (offset + j) * 5 + m) for j, v in enumerate(sentence["versions"][m])]
                )
            offset += n
        vectors = predict(model, kind, array, seqs, device, positions, mapped=True).reshape(
            -1, 5, 1024
        )
        return vectors, truth, donors
    selected = [g for g in groups()["val"] if g["word"] in bundled("vocabularies.json")["clinical"]]
    donors = [g["indices"] for g in selected]
    truth = [g["word"] for g in selected]
    if kind == "ours":
        # Original validation minibatches contain 25 groups / 125 individual windows.
        seqs = [
            [
                (i, j * 5 + m)
                for j, g in enumerate(selected[start : start + 25], start)
                for m, i in enumerate(g["indices"])
            ]
            for start in range(0, len(selected), 25)
        ]
        vectors = predict(model, kind, array, seqs, device, positions, True).reshape(-1, 5, 1024)
    else:
        all_vectors = predict(model, kind, array, array.sequences, device, positions)
        vectors = all_vectors[np.asarray(donors)]
    return vectors, truth, donors


NATURAL_CONTROLS = {"joint", "single_word", "shared_pulses", "independent_pulses", "timing"}


def validation_protocol(kind):
    if kind in NATURAL_CONTROLS:
        return dict(vocabulary=bundled("vocabularies.json")["natural"], observations=1,
                    inputs="natural_sentences", metric="top1_balanced_accuracy")
    return dict(vocabulary=bundled("vocabularies.json")["clinical"], observations=5,
                inputs="stitched" if kind == "stitched" else "word_groups",
                metric="top1_balanced_accuracy")


def validation_score(model, kind, array, device, positions, candidates, allowed):
    if kind in NATURAL_CONTROLS:
        # Keep every word as context; restrict only the scored targets and
        # retrieval vocabulary, exactly as in natural_evaluation().
        vectors = predict(model, kind, array, array.sequences, device, positions)
        truth = [item["word"] for item in array.items]
    else:
        vectors, truth, _ = validation_vectors(model, kind, array, device, positions)
        vectors = F.normalize(vectors.mean(1), dim=-1)
    selected = [i for i, word in enumerate(truth) if word in allowed]
    guesses = (vectors[selected] @ candidates.T).argmax(-1)
    return metrics([truth[i] for i in selected], [allowed[i] for i in guesses])[
        "top1_balanced_accuracy"]


def atomic_torch(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def run(work, kind, seed, device="cuda"):
    """Train with validation-based early stopping; defaults are 30 epochs, patience 10."""
    work = Path(work)
    out = work / "models" / kind / str(seed)
    out.mkdir(parents=True, exist_ok=True)
    cfg = bundled("experiment.json")
    import copy
    config = copy.deepcopy(cfg["models"][kind])
    protocol = validation_protocol(kind)
    if kind in NATURAL_CONTROLS and (out / "config.json").exists():
        previous = read(out / "config.json")
        if previous.get("validation") != protocol:
            raise ValueError("Validation protocol changed; use a fresh run directory. "
                             "Old best checkpoints cannot be compared with the new metric.")
    if (out / "complete.json").exists():
        return read(out / "complete.json")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    array = Windows(work, "train", kind)
    val = Windows(work, "val", kind)
    timing = None
    if kind == "timing":
        v = array[list(range(len(array)))]
        v = np.log(v[np.isfinite(v)])
        timing = dict(
            log_mean=float(v.mean(dtype=np.float64)), log_std=float(v.astype("float64").std())
        )
        # Statistics are fitted in float64 to the published (unrounded) intervals.
        raw = np.asarray(
            [array.annotations[i["record"]][str(i["event"])] for i in array.items], dtype="float64"
        )
        raw = np.log(raw[np.isfinite(raw)])
        timing = dict(log_mean=float(raw.mean()), log_std=float(raw.std()))
    model = build(kind, config, device, timing)
    target = np.load(DATA / "targets.npz")
    words = target["words"].tolist()
    lookup = {w: i for i, w in enumerate(words)}
    emb = torch.from_numpy(target["embeddings"]).float().to(device)
    positions = torch.from_numpy(np.load(DATA / "positions.npy")).float().to(device)[None]
    train_order = [i for g in groups()["train"] for i in g["indices"]]
    templates = bundled("stitched_train.json.gz") if kind == "stitched" else None
    warmup = (
        train_order[:2]
        if kind == "ours"
        else [0, 1]
        if kind == "joint"
        else templates["fallback"][0]
        if templates
        else array.sequences[0]
    )
    with torch.no_grad():
        forward(model, kind, torch.from_numpy(array[warmup]).to(device), positions, [len(warmup)])
    criterion = SigLipLoss(**config["loss"]).to(device)
    parameters = [
        p
        for p in model.parameters()
        if not isinstance(p, torch.nn.parameter.UninitializedParameter)
    ]
    optimizer = torch.optim.AdamW(
        parameters + list(criterion.parameters()), lr=config["lr"], weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    identity = dict(
        model=kind,
        seed=seed,
        config=config,
        timing=timing,
        cnn=cfg["cnn"],
        transformer=cfg["transformer"],
    )
    if kind in NATURAL_CONTROLS:
        identity["validation"] = protocol
    write(out / "config.json", identity)
    epoch = cursor = bad = 0
    best = -float("inf")
    history = []
    loss_sum = loss_count = 0
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=device, weights_only=False)
        assert saved["identity"] == identity, "Configuration changed since checkpoint"
        model.load_state_dict(saved["model"])
        criterion.load_state_dict(saved["criterion"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, cursor, bad, best, history, loss_sum, loss_count = [
            saved[k]
            for k in ["epoch", "cursor", "bad", "best", "history", "loss_sum", "loss_count"]
        ]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])

    def save(e, c):
        atomic_torch(
            out / "last.pt",
            dict(
                identity=identity,
                model=model.state_dict(),
                criterion=criterion.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=e,
                cursor=c,
                bad=bad,
                best=best,
                history=history,
                loss_sum=loss_sum,
                loss_count=loss_count,
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            ),
        )

    allowed = protocol["vocabulary"]
    candidates = F.normalize(emb[[lookup[w] for w in allowed]], dim=-1).cpu()
    for e in range(epoch, config["max_epochs"]):
        if bad >= config["patience"]:
            break
        if kind == "ours":
            order = np.random.default_rng(seed + e).permutation(len(train_order))
            packed = [
                [[train_order[j] for j in order[s : s + 128]]] for s in range(0, len(order), 128)
            ]
            if len(packed[-1][0]) == 1:
                packed[-2][0] += packed.pop()[0]
        else:
            if kind == "stitched":
                from .stitching import schedule, DATA_SEED

                sequences, h, _ = schedule(
                    bank("train"), templates["templates"], templates["fallback"], e
                )
                order = np.random.default_rng(DATA_SEED + 20000 + e).permutation(len(sequences))
                write(out / f"schedule_{e + 1}.json", dict(sha256=h))
            else:
                sequences = array.sequences
                order = np.random.default_rng(seed + e).permutation(len(sequences))
            packed = list(batches([sequences[i] for i in order], 128))
        model.train()
        criterion.train()
        last_save = time.monotonic()
        for step in range(cursor, len(packed)):
            seqs = packed[step]
            ids = [i for seq in seqs for i in seq]
            x = torch.from_numpy(array[ids]).to(device)
            labels = [lookup[array.items[i]["word"]] for i in ids]
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model, kind, x, positions, [len(s) for s in seqs])
            loss = criterion(pred, emb[labels])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            loss_count += 1
            if (step + 1) % 50 == 0 or time.monotonic() - last_save >= 120:
                save(e, step + 1)
                last_save = time.monotonic()
                status = dict(
                    state="training",
                    epoch=e + 1,
                    step=step + 1,
                    batches=len(packed),
                    loss=loss_sum / loss_count,
                    bad_epochs=bad,
                )
                write(out / "status.json", status)
                print(status, flush=True)
        score = validation_score(model, kind, val, device, positions, candidates, allowed)
        if score > best:
            best = score
            bad = 0
            atomic_torch(
                out / "best.pt",
                dict(
                    identity=identity,
                    model=model.state_dict(),
                    epoch=e + 1,
                    validation_balanced_accuracy=score,
                ),
            )
        else:
            bad += 1
        history.append(
            dict(
                epoch=e + 1,
                loss=loss_sum / loss_count,
                validation_balanced_accuracy=score,
                best=best,
                bad_epochs=bad,
            )
        )
        write(out / "history.json", history)
        print(dict(epoch=e + 1, validation_balanced_accuracy=score, best=best, bad_epochs=bad), flush=True)
        scheduler.step()
        cursor = 0
        loss_sum = loss_count = 0
        save(e + 1, 0)
    result = dict(state="complete", epochs=len(history), best_validation=best, bad_epochs=bad)
    write(out / "complete.json", result)
    return result


def load_model(work, kind, seed, device="cuda"):
    saved = torch.load(checkpoint(work, kind, seed), map_location="cpu", weights_only=False)
    cfg = saved["identity"]
    model = build(kind, cfg["config"], device, cfg.get("timing"))
    model.load_state_dict(saved["model"])
    model.eval()
    positions = torch.from_numpy(np.load(DATA / "positions.npy")).float().to(device)[None]
    return model, positions, saved
