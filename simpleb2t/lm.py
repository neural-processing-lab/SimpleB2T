"""Frozen full-vocabulary Qwen likelihoods. Generated prefixes only during decoding."""

import collections, json, pickle
from pathlib import Path
import numpy as np

PREFIX = "A patient communicates a short request to hospital staff.\nPatient:"


def log_probabilities(scores, temperature):
    values = np.asarray(scores, dtype=np.float64) / temperature
    values -= values.max(axis=-1, keepdims=True)
    return values - np.log(np.exp(values).sum(axis=-1, keepdims=True))


class QwenScorer:
    def __init__(self, model_path, words, cache_path, prompt=PREFIX, device="cuda", revision=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.torch = torch
        self.words = words
        self.cache_path = Path(cache_path)
        self.prompt = prompt
        self.device = device
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            torch_dtype=torch.float32,
            device_map={"": device},
            attn_implementation="sdpa",
        ).eval()
        self.base = tuple(self.tokenizer.encode(self.prompt, add_special_tokens=False))
        self.pieces = {
            initial: [
                tuple(
                    self.tokenizer.encode(" " + self.surface(w, initial), add_special_tokens=False)
                )
                for w in words
            ]
            for initial in (False, True)
        }
        self.period = tuple(self.tokenizer.encode(".", add_special_tokens=False))
        assert len(self.period) == 1
        self.question = tuple(self.tokenizer.encode("?", add_special_tokens=False))
        assert len(self.question) == 1 and self.question != self.period
        self.terminal_tokens = self.period + self.question
        identity = dict(
            model_path=str(model_path),
            revision=revision,
            words=words,
            prefix=self.prompt,
            terminal=[".", "?"],
            precision="FP32",
        )
        self.cache = {"word": {}, "end": {}, "identity": identity}
        if self.cache_path.exists():
            with self.cache_path.open("rb") as stream:
                self.cache = pickle.load(stream)
            assert self.cache["identity"] == identity
        self.check_token_boundaries()
        self.check_likelihoods()

    @staticmethod
    def surface(word, initial):
        return "I" if word == "i" else word.capitalize() if initial else word

    def tokens(self, prefix):
        ids = self.base
        for pos, index in enumerate(prefix):
            ids += self.pieces[pos == 0][index]
        return ids

    def check_token_boundaries(self):
        # Verify exact concatenation for every possible initial and adjacent
        # vocabulary word, including contractions and fixed terminal punctuation.
        for i, w in enumerate(self.words):
            text = self.prompt + " " + self.surface(w, True)
            ids = self.base + self.pieces[True][i]
            assert tuple(self.tokenizer.encode(text, add_special_tokens=False)) == ids
            assert (
                tuple(self.tokenizer.encode(text + ".", add_special_tokens=False))
                == ids + self.period
            )
            assert (
                tuple(self.tokenizer.encode(text + "?", add_special_tokens=False))
                == ids + self.question
            )
            for j, v in enumerate(self.words):
                full = text + " " + self.surface(v, False)
                assert (
                    tuple(self.tokenizer.encode(full, add_special_tokens=False))
                    == ids + self.pieces[False][j]
                )

    def check_likelihoods(self):
        torch = self.torch
        # Compare the branching scorer against an independent teacher-forced
        # forward pass, for an initial word and a multi-token continuation.
        for prefix, word in [
            ((), "don't"),
            ((self.words.index("can"), self.words.index("you")), "don't"),
        ]:
            index = self.words.index(word)
            base = self.tokens(prefix)
            pieces = self.pieces[not prefix][index]
            actual = float(self.score_words([prefix])[0, index])
            with torch.inference_mode():
                ids = torch.tensor([base + pieces], device=self.device)
                logits = self.model(input_ids=ids, use_cache=False).logits[0].float()
                logp = torch.log_softmax(
                    logits[len(base) - 1 : len(base) + len(pieces) - 1], dim=-1
                )
                expected = float(
                    logp[
                        torch.arange(len(pieces), device=self.device),
                        torch.tensor(pieces, device=self.device),
                    ].sum()
                )
            assert abs(actual - expected) < 0.001, (actual, expected)
            print(
                json.dumps(
                    dict(
                        check="teacher_forced_likelihood",
                        word=word,
                        prefix=list(prefix),
                        branching=actual,
                        teacher_forced=expected,
                        absolute_error=abs(actual - expected),
                    )
                ),
                flush=True,
            )

    def query(self, requests):
        """Return exact next-token log probabilities from the full vocabulary."""
        torch = self.torch
        result = {}
        keys = list(requests)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        with torch.inference_mode():
            for start in range(0, len(keys), 32):
                selected = keys[start : start + 32]
                length = max(map(len, selected))
                ids = torch.full((len(selected), length), pad, device=self.device, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for row, tokens in enumerate(selected):
                    ids[row, -len(tokens) :] = torch.tensor(tokens, device=self.device)
                    mask[row, -len(tokens) :] = 1
                positions = (mask.cumsum(-1) - 1).clamp_min(0)
                logits = (
                    self.model(
                        input_ids=ids,
                        attention_mask=mask,
                        position_ids=positions,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    .logits[:, 0]
                    .float()
                )
                normalizer = torch.logsumexp(logits, dim=-1)
                for row, key in enumerate(selected):
                    wanted = sorted(requests[key])
                    values = (logits[row, wanted] - normalizer[row]).cpu().numpy()
                    result[key] = dict(zip(wanted, map(float, values)))
        return result

    def score_words(self, prefixes):
        missing = [p for p in prefixes if p not in self.cache["word"]]
        requests = collections.defaultdict(set)
        for prefix in missing:
            base = self.tokens(prefix)
            for pieces in self.pieces[not prefix]:
                for i, token in enumerate(pieces):
                    requests[base + pieces[:i]].add(token)
        if requests:
            scores = self.query(requests)
            for prefix in missing:
                base = self.tokens(prefix)
                self.cache["word"][prefix] = np.asarray(
                    [
                        sum(scores[base + pieces[:i]][token] for i, token in enumerate(pieces))
                        for pieces in self.pieces[not prefix]
                    ],
                    dtype=np.float32,
                )
        return np.stack([self.cache["word"][p] for p in prefixes])

    def score_end(self, prefixes):
        missing = [p for p in prefixes if p not in self.cache["end"]]
        if missing:
            scores = self.query({self.tokens(p): set(self.terminal_tokens) for p in missing})
            for p in missing:
                values = [scores[self.tokens(p)][t] for t in self.terminal_tokens]
                self.cache["end"][p] = float(np.logaddexp(*values))
        return np.asarray([self.cache["end"][p] for p in prefixes])

    def continuation_likelihood(self, bases, pieces):
        """Score an observed suffix for a diagnostic (never called by beam search)."""
        if not pieces:
            return np.zeros(len(bases))
        torch = self.torch
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        values = []
        for start in range(0, len(bases), 46):
            sequences = [b + pieces[:-1] for b in bases[start : start + 46]]
            length = max(map(len, sequences))
            ids = torch.full((len(sequences), length), pad, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for row, sequence in enumerate(sequences):
                ids[row, -len(sequence) :] = torch.tensor(sequence, device=self.device)
                mask[row, -len(sequence) :] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
                    use_cache=False,
                    logits_to_keep=len(pieces),
                ).logits.float()
                target = torch.tensor(pieces, device=self.device)[None, :, None].expand(
                    len(sequences), -1, -1
                )
                logp = logits.gather(-1, target).squeeze(-1) - torch.logsumexp(logits, dim=-1)
                values.extend(logp.sum(-1).cpu().tolist())
                del logits, logp
        return np.asarray(values, dtype=float)

    def save(self):
        temp = self.cache_path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            pickle.dump(self.cache, stream, protocol=5)
        temp.replace(self.cache_path)


def nbest(brain, lm, weight, beam=50):
    # Identical expansion/pruning/tie order and terminal score to beam_search.
    prefixes = [()]
    scores = np.zeros(1)
    for row in brain:
        expanded = scores[:, None] + row[None, :] + weight * np.asarray(lm.score_words(prefixes))
        order = np.argsort(-expanded.ravel(), kind="stable")[:beam]
        width = brain.shape[1]
        prefixes = [prefixes[i // width] + (int(i % width),) for i in order]
        scores = expanded.ravel()[order]
    scores += weight * np.asarray(lm.score_end(prefixes))
    order = np.argsort(-scores, kind="stable")
    return [(prefixes[i], float(scores[i])) for i in order]


PROMPTS = {
    "A": "Sentence:",
    "B": "The following is a sentence from an English passage:",
    "C": "The following is a sentence from everyday speech:",
    "D": "The following is a sentence spoken in a clinical setting:",
    "E": PREFIX,
    "F": "A patient who cannot speak communicates a short message to hospital staff about comfort, positioning, personal care, their surroundings, or interaction with other people.\nPatient:",
}
