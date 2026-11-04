"""Feature extraction for domain names: normalize_domain(), FEATURE_NAMES, extract_features().

This is the single source of truth for features. Every other module imports from here.

All features are computed on the second-level label only (mail.google.co.uk -> "google").
The n-gram tables and the dictionary come from the committed English word list
(data/reference/english_words.txt), never from training data. They are built once, at import.
"""

import math
import re
from collections import Counter
from pathlib import Path

import tldextract

from src.config import WORDLIST_PATH

FEATURE_NAMES: list[str] = [
    "length",  # characters in the label
    "digit_ratio",  # digits / length
    "vowel_ratio",  # vowels (aeiou) / letters; 0 if no letters
    "hyphen_count",  # number of '-'
    "entropy",  # Shannon entropy of the character distribution, in bits
    "unique_char_ratio",  # distinct characters / length
    "longest_consonant_run",  # longest run of consecutive consonants (y counts as consonant)
    "digit_letter_transitions",  # adjacent letter/digit pairs, either order ("a1b2" -> 3)
    "bigram_mean_logfreq",  # mean log10 P(bigram) under the word list, add-one smoothed
    "trigram_mean_logfreq",  # same for trigrams
    "bigram_unseen_ratio",  # share of bigrams that never occur in the word list
    "dict_coverage",  # share of letters covered by the best split into dictionary words
    "dict_word_count",  # number of words in that split
    "longest_dict_word_ratio",  # longest dictionary word inside the label / length
    "is_dict_word",  # 1 if the whole label is a dictionary word
]

VOWELS = frozenset("aeiou")
LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
DIGITS = frozenset("0123456789")
MIN_SPLIT_WORD_LEN = 3  # shortest dictionary word the split may use

# Offline extractor: bundled public suffix list, no network, no disk cache.
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://")
_NON_LETTERS = re.compile(r"[^a-z]+")


# ---------------------------------------------------------------------------
# Normalization and label extraction
# ---------------------------------------------------------------------------


def normalize_domain(raw: str) -> str:
    """Trim, lowercase, and strip scheme, user info, path/query/fragment, port and trailing dot."""
    text = raw.strip().lower()
    text = _SCHEME.sub("", text)
    text = re.split(r"[/?#]", text, maxsplit=1)[0]  # path, query, fragment
    text = text.rsplit("@", 1)[-1]  # user info, e.g. user:pass@host
    text = text.split(":", 1)[0]  # port
    return text.rstrip(".")


def second_level_label(domain: str) -> str:
    """Return the registrable label of a domain, e.g. 'mail.google.co.uk' -> 'google'."""
    return _EXTRACTOR(normalize_domain(domain)).domain


# ---------------------------------------------------------------------------
# Reference data: word list and n-gram tables
# ---------------------------------------------------------------------------


def load_wordlist(path: Path = WORDLIST_PATH) -> frozenset[str]:
    """Read one word per line (already cleaned by scripts/build_wordlist.py)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip())


def ngrams(text: str, n: int) -> list[str]:
    """All overlapping n-character substrings of text."""
    return [text[i : i + n] for i in range(len(text) - n + 1)]


class NgramTable:
    """Add-one smoothed log10 probabilities of letter n-grams, counted over a word list.

    Each word in the list counts once (word types, not word frequencies). The vocabulary is all
    26**n letter n-grams, so an n-gram that never appears still gets a small probability.
    """

    def __init__(self, words: frozenset[str], n: int) -> None:
        self.n = n
        self.counts = Counter(gram for word in words for gram in ngrams(word, n))
        self.denominator = sum(self.counts.values()) + 26**n
        self.unseen_logp = math.log10(1 / self.denominator)

    def logp(self, gram: str) -> float:
        """log10 probability of one n-gram (n-grams with non-letters count as unseen)."""
        return math.log10((self.counts.get(gram, 0) + 1) / self.denominator)

    def mean_logp(self, label: str) -> float:
        """Mean log10 probability over the label's n-grams; unseen value if it has none."""
        grams = ngrams(label, self.n)
        if not grams:
            return self.unseen_logp
        return sum(self.logp(g) for g in grams) / len(grams)

    def unseen_ratio(self, label: str) -> float:
        """Share of the label's n-grams never seen in the word list; 1.0 if it has none."""
        grams = ngrams(label, self.n)
        if not grams:
            return 1.0
        return sum(1 for g in grams if g not in self.counts) / len(grams)


WORDS: frozenset[str] = load_wordlist()
MAX_SPLIT_WORD_LEN: int = max(len(w) for w in WORDS)
BIGRAMS = NgramTable(WORDS, 2)
TRIGRAMS = NgramTable(WORDS, 3)


# ---------------------------------------------------------------------------
# Dictionary split
# ---------------------------------------------------------------------------


def split_into_words(
    label: str,
    words: frozenset[str] = WORDS,
    min_len: int = MIN_SPLIT_WORD_LEN,
    max_len: int = MAX_SPLIT_WORD_LEN,
) -> tuple[list[str], int]:
    """Split a label into dictionary words, covering as many letters as possible.

    Non-letters (digits, hyphens) separate the label into parts. Each part is split with dynamic
    programming: best[i] is the best split of the first i letters, scored as
    (letters covered, -words used), so ties go to the split with fewer words.
    Returns (words used in order, length of the longest dictionary word found anywhere).
    """
    found: list[str] = []
    longest = 0
    for part in _NON_LETTERS.split(label):
        n = len(part)
        # best[i] = (covered letters, -number of words, words used)
        best: list[tuple[int, int, list[str]]] = [(0, 0, [])] * (n + 1)
        for end in range(1, n + 1):
            best[end] = best[end - 1]  # option 1: letter end-1 is left uncovered
            for start in range(max(0, end - max_len), end - min_len + 1):
                piece = part[start:end]
                if piece in words:
                    longest = max(longest, len(piece))
                    covered, neg_count, used = best[start]
                    candidate = (covered + len(piece), neg_count - 1, used + [piece])
                    if candidate[:2] > best[end][:2]:
                        best[end] = candidate
        found.extend(best[n][2])
    return found, longest


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def shannon_entropy(text: str) -> float:
    """Shannon entropy in bits of the character distribution of text (0 for empty text)."""
    total = len(text)
    return sum((c / total) * math.log2(total / c) for c in Counter(text).values())


def longest_run(text: str, chars: frozenset[str]) -> int:
    """Length of the longest run of consecutive characters that are all in chars."""
    longest = current = 0
    for ch in text:
        current = current + 1 if ch in chars else 0
        longest = max(longest, current)
    return longest


def label_features(label: str) -> list[float]:
    """Compute FEATURE_NAMES (in order) for an already-extracted second-level label."""
    if not label:
        raise ValueError("Empty second-level label: cannot extract features")

    length = len(label)
    letters = [ch for ch in label if ch in LETTERS]
    vowels = sum(1 for ch in letters if ch in VOWELS)
    digits = sum(1 for ch in label if ch in DIGITS)
    transitions = sum(
        1
        for a, b in zip(label, label[1:], strict=False)
        if (a in LETTERS and b in DIGITS) or (a in DIGITS and b in LETTERS)
    )
    split, longest_word = split_into_words(label)
    covered = sum(len(w) for w in split)

    values = {
        "length": length,
        "digit_ratio": digits / length,
        "vowel_ratio": vowels / len(letters) if letters else 0.0,
        "hyphen_count": label.count("-"),
        "entropy": shannon_entropy(label),
        "unique_char_ratio": len(set(label)) / length,
        "longest_consonant_run": longest_run(label, LETTERS - VOWELS),
        "digit_letter_transitions": transitions,
        "bigram_mean_logfreq": BIGRAMS.mean_logp(label),
        "trigram_mean_logfreq": TRIGRAMS.mean_logp(label),
        "bigram_unseen_ratio": BIGRAMS.unseen_ratio(label),
        "dict_coverage": covered / len(letters) if letters else 0.0,
        "dict_word_count": len(split),
        "longest_dict_word_ratio": longest_word / length,
        "is_dict_word": 1 if label in WORDS else 0,
    }
    return [float(values[name]) for name in FEATURE_NAMES]


def extract_features(domain: str) -> list[float]:
    """Normalize a raw domain, take its second-level label, and return its features.

    The output has the same order as FEATURE_NAMES. Raises ValueError if the domain has no
    second-level label (for example an empty string or a bare suffix like 'co.uk').
    """
    return label_features(second_level_label(domain))
