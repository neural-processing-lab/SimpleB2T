"""Check fixed inputs, or independently regenerate the frozen target embeddings."""

import numpy as np
from .io import DATA, bundled, sha


def verify():
    manifest = bundled("checksums.json")
    for filename, expected in manifest.items():
        assert sha(DATA / filename) == expected, filename
    natural = bundled("natural.json.gz")
    groups = bundled("groups.json.gz")
    vocab = bundled("vocabularies.json")
    recordings = bundled("recordings.json.gz")
    used = {}
    split_records = {}
    for split, data in natural.items():
        flat = [i for s in data["sequences"] for i in s]
        assert sorted(flat) == list(range(len(data["items"])))
        split_records[split] = {i["record"] for i in data["items"]}
        for i, item in enumerate(data["items"]):
            key = (item["record"], item["event"])
            assert key not in used
            used[key] = (split, i)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not split_records[a] & split_records[b]
        assert not (
            {recordings[r]["session"] for r in split_records[a]}
            & {recordings[r]["session"] for r in split_records[b]}
        )
    for split in ["train", "val", "test"]:
        selected = []
        for g in groups[split]:
            assert len(g["indices"]) == 5
            for i in g["indices"]:
                assert natural[split]["items"][i]["word"] == g["word"]
            selected.extend(g["indices"])
        assert len(selected) == len(set(selected)), f"Reused {split} occurrence"
    assert len(groups["test"]) == 1275 and len({g["sentence_id"] for g in groups["test"]}) == 200
    assert sum(g["sentence_id"] <= 100 for g in groups["test"]) == 595
    assert len(vocab["clinical"]) == 92 and len(vocab["natural"]) == 50
    assert len(groups["dev"]) == 307 and len({g["sentence_id"] for g in groups["dev"]}) == 50
    for group in groups["dev"]:
        assert len(group["indices"]) == 5
        assert all(natural["val"]["items"][i]["word"] == group["word"] for i in group["indices"])
    from .data import bank

    for split, definition, key in [
        ("train", bundled("stitched_train.json.gz"), "fallback"),
        ("test", bundled("stitched_test.json.gz"), "sentences"),
    ]:
        b = bank(split)
        seqs = definition[key] if split == "train" else [r["donors"] for r in definition[key]]
        assert all(b.valid(s) for s in seqs)
        flat = [i for s in seqs for i in s]
        assert len(flat) == len(set(flat))
    validation_bank = bank("val")
    for s in bundled("stitched_val.json.gz")["sentences"]:
        assert all(validation_bank.valid(v) for v in s["versions"])
    print(
        "PASS: checksums, session isolation, occurrence uniqueness, 200 clinical sentences, 50 development sentences, stitched non-overlap."
    )


def regenerate_targets(output, device):
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    model_id = "google-t5/t5-large"
    revision = "150ebc2c4b72291e770f58e6057481c8d2ed331a"
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, truncation_side="left")
    model = T5EncoderModel.from_pretrained(model_id, revision=revision).to(device).eval()
    original = np.load(DATA / "targets.npz")
    words = original["words"].tolist()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(words), 32):
            inputs = tok(
                words[start : start + 32],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            states = model(**inputs, output_hidden_states=True).hidden_states
            index = int(0.5 * len(states) - 1e-6)
            assert index == 12
            for h, mask in zip(states[index], inputs["attention_mask"]):
                parts.append(h[mask.bool()].mean(0).cpu().numpy())
    result = np.stack(parts)
    np.savez_compressed(output, words=np.asarray(words), embeddings=result)
    print("Maximum absolute difference:", float(np.max(abs(result - original["embeddings"]))))
    np.testing.assert_allclose(result, original["embeddings"], atol=1e-3, rtol=1e-3)
