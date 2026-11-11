"""A small linear classifier for telling two sets of short chat replies apart.

numpy only. It is used twice: to measure how easily the bot's replies can be
told from mine (the lower the better), and to estimate when I would speak.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import numpy as np

NGRAM_DIM = 4096
# Per-reply statistics, in the order `text_stats` returns them.
STAT_NAMES = (
    "chars_per_message",
    "longest_message",
    "messages",
    "ends_with_period",
    "ends_with_mark",
    "commas_per_char",
    "emoji_per_char",
    "latin_per_char",
    "digits_per_char",
    "has_question",
    "repeated_chars",
    "spaces_per_char",
)
_PERIODS = "。."
_MARKS = "！!？?~～"
_COMMAS = "，,、"


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return code >= 0x1F000 or 0x2600 <= code <= 0x27BF


def text_stats(messages: Sequence[str]) -> list[float]:
    """Shape of one reply (a burst of messages), with no reference to its words."""
    messages = [m for m in messages if m]
    if not messages:
        return [0.0] * len(STAT_NAMES)
    text = "".join(messages)
    chars = max(len(text), 1)
    count = len(messages)
    repeated = sum(1 for m in messages for a, b in zip(m, m[1:]) if a == b)
    return [
        len(text) / count,
        float(max(len(m) for m in messages)),
        float(count),
        sum(m[-1] in _PERIODS for m in messages) / count,
        sum(m[-1] in _MARKS for m in messages) / count,
        sum(c in _COMMAS for c in text) / chars,
        sum(_is_emoji(c) for c in text) / chars,
        sum(c.isascii() and c.isalpha() for c in text) / chars,
        sum(c.isdigit() for c in text) / chars,
        float(any("?" in m or "？" in m for m in messages)),
        repeated / chars,
        sum(c.isspace() for c in text) / chars,
    ]


def ngrams(messages: Sequence[str]) -> list[str]:
    """Character 1- to 3-grams of each message, with markers for its two ends."""
    grams: list[str] = []
    for message in messages:
        text = f"^{message}$"
        for size in (1, 2, 3):
            grams.extend(text[i : i + size] for i in range(len(text) - size + 1))
    return grams


def bucket(gram: str) -> int:
    """Stable across runs, unlike the built-in hash."""
    return zlib.crc32(gram.encode("utf-8")) % NGRAM_DIM


def featurize(samples: Sequence[Sequence[str]]) -> np.ndarray:
    """One row per reply: hashed character n-grams, then the reply statistics."""
    out = np.zeros((len(samples), NGRAM_DIM + len(STAT_NAMES)), dtype=np.float32)
    for row, messages in enumerate(samples):
        for gram in ngrams(messages):
            out[row, bucket(gram)] += 1.0
        counts = out[row, :NGRAM_DIM]
        np.log1p(counts, out=counts)
        norm = float(np.linalg.norm(counts))
        if norm > 0:
            counts /= norm
        out[row, NGRAM_DIM:] = text_stats(messages)
    return out


class LogReg:
    """L2-regularised logistic regression fitted by gradient descent.

    The last `scaled` columns are standardised on the training data; the rest
    are used as given.
    """

    def __init__(self, l2: float = 1.0, steps: int = 400, scaled: int = 0) -> None:
        self.l2 = l2
        self.steps = steps
        self.scaled = scaled
        self.weights = np.zeros(0, dtype=np.float64)
        self.bias = 0.0
        self.mean = np.zeros(0)
        self.std = np.ones(0)

    def _scale(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self.scaled:
            return x
        x = x.copy()
        tail = (x[:, -self.scaled :] - self.mean) / self.std
        # Clipped and shrunk so a few wild statistics cannot outvote everything else.
        x[:, -self.scaled :] = np.clip(tail, -3.0, 3.0) / np.sqrt(self.scaled)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogReg:
        y = np.asarray(y, dtype=np.float64)
        if self.scaled:
            tail = np.asarray(x, dtype=np.float64)[:, -self.scaled :]
            self.mean = tail.mean(axis=0)
            self.std = np.maximum(tail.std(axis=0), 1e-6)
        x = self._scale(x)
        n, d = x.shape
        b = float(np.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6)))
        penalty = self.l2 / n
        step = 1.0 / (0.25 * float((x * x).sum(axis=1).max() + 1.0) + penalty)
        if n <= d:
            # Few samples, many features: the same descent written over the samples
            # (weights = x.T @ a), so each step costs n*n instead of n*d.
            gram = x @ x.T
            a = np.zeros(n)
            for _ in range(self.steps):
                error = 1.0 / (1.0 + np.exp(-(gram @ a + b))) - y
                a = (1.0 - step * penalty) * a - step * error / n
                b -= step * float(error.mean())
            w = x.T @ a
        else:
            w = np.zeros(d)
            for _ in range(self.steps):
                error = 1.0 / (1.0 + np.exp(-(x @ w + b))) - y
                w -= step * (x.T @ error / n + penalty * w)
                b -= step * float(error.mean())
        self.weights, self.bias = w, b
        return self

    def decision(self, x: np.ndarray) -> np.ndarray:
        return self._scale(x) @ self.weights + self.bias

    def probability(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision(x)))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a positive scores above a negative; ties count half."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(scores) + 1):  # average the ranks of tied scores
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def fold_of(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign each group to a fold, so samples sharing a group are never split."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    fold_by_group = {group: int(shuffled[i] % folds) for i, group in enumerate(unique)}
    return np.array([fold_by_group[group] for group in groups])


def cross_val_scores(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int = 5,
    seed: int = 0,
    l2: float = 1.0,
    scaled: int = 0,
) -> np.ndarray:
    """A score for every sample from a model that never saw its group."""
    fold = fold_of(groups, folds, seed)
    scores = np.zeros(len(y), dtype=np.float64)
    for index in range(folds):
        held = fold == index
        if not held.any() or held.all():
            continue
        model = LogReg(l2=l2, scaled=scaled).fit(x[~held], y[~held])
        scores[held] = model.decision(x[held])
    return scores


def auc_interval(
    scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, rounds: int = 400, seed: int = 0
) -> tuple[float, float]:
    """A 95% interval for the AUC, resampling whole groups."""
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(rounds):
        picked = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        values.append(auc(scores[picked], labels[picked]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
