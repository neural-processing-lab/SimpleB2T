"""Corpus WER, exact sentence match, and macro word accuracy."""

import collections


def word_edit_distance(reference, hypothesis):
    """Standard token Levenshtein distance, allowing S/D/I alignment."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def metrics(words, predicted, groups=None):
    assert len(words) == len(predicted) and len(words)
    totals, correct = collections.Counter(words), collections.Counter()
    for word, guess in zip(words, predicted):
        correct[word] += word == guess
    per_word = {
        w: {"n": n, "correct": correct[w], "accuracy": correct[w] / n}
        for w, n in sorted(totals.items())
    }
    result = dict(
        top1_balanced_accuracy=sum(x["accuracy"] for x in per_word.values()) / len(per_word),
        top1_word_accuracy=sum(correct.values()) / len(words),
        examples=len(words),
        observed_word_types=len(totals),
        per_word=per_word,
    )
    if groups is not None:
        sentences = collections.defaultdict(list)
        for group, word, guess in zip(groups, words, predicted):
            sentences[group["sentence_id"]].append((group["word_position"], word, guess))
        rows = []
        for sid, entries in sorted(sentences.items()):
            entries.sort()
            reference = [x[1] for x in entries]
            hypothesis = [x[2] for x in entries]
            edits = word_edit_distance(reference, hypothesis)
            rows.append(
                dict(
                    sentence_id=sid,
                    reference=" ".join(reference),
                    prediction=" ".join(hypothesis),
                    reference_words=len(reference),
                    word_edits=edits,
                    wer=edits / len(reference),
                )
            )
        result.update(
            word_error_rate=sum(r["word_edits"] for r in rows) / len(words),
            word_edits=sum(r["word_edits"] for r in rows),
            position_error_rate=1 - result["top1_word_accuracy"],
            sentence_exact_accuracy=sum(r["word_edits"] == 0 for r in rows) / len(rows),
            sentence_predictions=rows,
        )
    return result
