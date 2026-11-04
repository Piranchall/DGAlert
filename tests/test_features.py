"""Tests for src.features: normalization, label extraction, n-gram tables, word split, features.

Expected values are computed by hand in the comments next to each assertion.
"""

import math

import pytest

from src import features
from src.features import (
    FEATURE_NAMES,
    NgramTable,
    extract_features,
    normalize_domain,
    second_level_label,
    split_into_words,
)


def feats(domain: str) -> dict[str, float]:
    """Features of a domain as a {name: value} dict, for readable assertions."""
    return dict(zip(FEATURE_NAMES, extract_features(domain), strict=True))


# --- normalize_domain -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Google.COM  ", "google.com"),
        ("https://www.example.com/path?q=1#frag", "www.example.com"),
        ("http://user:pass@example.com:8080/x", "example.com"),
        ("ftp://files.example.org", "files.example.org"),
        ("EXAMPLE.com:443", "example.com"),
        ("example.com.", "example.com"),
        ("example.com?ref=1", "example.com"),
        ("google.com", "google.com"),
    ],
)
def test_normalize_domain(raw: str, expected: str) -> None:
    assert normalize_domain(raw) == expected


# --- second_level_label -----------------------------------------------------


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("mail.google.co.uk", "google"),  # subdomain + multi-part suffix
        ("www.bbc.co.uk", "bbc"),
        ("a.b.c.example.com", "example"),  # deep subdomains
        ("shop.example.com.au", "example"),
        ("HTTPS://Mail.Google.co.uk:443/inbox", "google"),  # normalized first
        ("localhost", "localhost"),  # no suffix at all
    ],
)
def test_second_level_label(domain: str, expected: str) -> None:
    assert second_level_label(domain) == expected


@pytest.mark.parametrize("domain", ["", "   ", "co.uk", "https://", "."])
def test_extract_features_rejects_domains_without_label(domain: str) -> None:
    with pytest.raises(ValueError):
        extract_features(domain)


# --- output shape and determinism -------------------------------------------


def test_feature_names_are_unique() -> None:
    assert len(FEATURE_NAMES) == 15
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


@pytest.mark.parametrize("domain", ["google.com", "pidfksipycmyg.com", "a1b2.com", "x.io"])
def test_output_length_and_type(domain: str) -> None:
    values = extract_features(domain)
    assert len(values) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) and math.isfinite(v) for v in values)


def test_extract_features_is_deterministic() -> None:
    assert extract_features("qxxsoxcwrcplo.net") == extract_features("qxxsoxcwrcplo.net")


def test_features_depend_only_on_second_level_label() -> None:
    # Scheme, case, subdomain, port, path and TLD must not change the features.
    assert extract_features("https://WWW.Google.co.uk:443/a?b") == extract_features("google.com")


def test_reference_data_is_not_reloaded_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Word list is loaded once at import; extract_features must not read it again."""

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_wordlist called during feature extraction")

    monkeypatch.setattr(features, "load_wordlist", fail)
    assert len(extract_features("subjectclean.net")) == len(FEATURE_NAMES)


# --- hand-computed simple features ------------------------------------------


def test_features_of_aaaa() -> None:
    f = feats("aaaa.com")
    assert f["length"] == 4
    assert f["entropy"] == 0.0  # one symbol with p = 1
    assert f["unique_char_ratio"] == 0.25  # 1 distinct / 4
    assert f["vowel_ratio"] == 1.0  # 4 vowels / 4 letters
    assert f["longest_consonant_run"] == 0


def test_features_of_a1b2() -> None:
    f = feats("a1b2.com")
    assert f["digit_ratio"] == 0.5  # 2 digits / 4
    assert f["digit_letter_transitions"] == 3  # a|1, 1|b, b|2
    assert f["entropy"] == pytest.approx(2.0)  # 4 distinct symbols, p = 1/4 each
    assert f["unique_char_ratio"] == 1.0
    assert f["vowel_ratio"] == 0.5  # 'a' / letters {a, b}
    assert f["longest_consonant_run"] == 1  # 'b'


def test_entropy_of_two_equal_symbols() -> None:
    assert feats("aabb.com")["entropy"] == pytest.approx(1.0)  # p = 1/2 each


def test_consonant_run_and_vowels() -> None:
    f = feats("bcdfga.com")
    assert f["longest_consonant_run"] == 5  # b c d f g
    assert f["vowel_ratio"] == pytest.approx(1 / 6)


def test_hyphen_count() -> None:
    assert feats("my-site-x.com")["hyphen_count"] == 2


def test_all_digit_label_has_no_letter_features() -> None:
    f = feats("123.com")
    assert f["digit_ratio"] == 1.0
    assert f["vowel_ratio"] == 0.0  # no letters
    assert f["dict_coverage"] == 0.0  # no letters
    assert f["digit_letter_transitions"] == 0


# --- n-gram table on a tiny word list ---------------------------------------
# words {"abc", "abd"}: bigrams ab x2, bc x1, bd x1 -> 4 bigrams; denominator 4 + 26**2 = 680


@pytest.fixture
def tiny_bigrams() -> NgramTable:
    return NgramTable(frozenset({"abc", "abd"}), 2)


def test_ngram_logp_add_one_smoothing(tiny_bigrams: NgramTable) -> None:
    assert tiny_bigrams.denominator == 680
    assert tiny_bigrams.logp("ab") == pytest.approx(math.log10(3 / 680))  # (2 + 1) / 680
    assert tiny_bigrams.logp("bc") == pytest.approx(math.log10(2 / 680))  # (1 + 1) / 680
    assert tiny_bigrams.logp("zz") == pytest.approx(math.log10(1 / 680))  # unseen
    assert tiny_bigrams.logp("a1") == tiny_bigrams.unseen_logp  # non-letters are unseen


def test_ngram_mean_and_unseen_ratio(tiny_bigrams: NgramTable) -> None:
    # "abc" -> ab, bc
    expected = (math.log10(3 / 680) + math.log10(2 / 680)) / 2
    assert tiny_bigrams.mean_logp("abc") == pytest.approx(expected)
    assert tiny_bigrams.unseen_ratio("abz") == 0.5  # ab seen, bz unseen
    # A 1-character label has no bigrams: unseen value, not 0 ("very common").
    assert tiny_bigrams.mean_logp("a") == tiny_bigrams.unseen_logp
    assert tiny_bigrams.unseen_ratio("a") == 1.0


def test_english_like_label_scores_higher_than_random() -> None:
    english, random_like = feats("stackoverflow.com"), feats("qxxsoxcwrcplo.net")
    assert english["bigram_mean_logfreq"] > random_like["bigram_mean_logfreq"]
    assert english["trigram_mean_logfreq"] > random_like["trigram_mean_logfreq"]


# --- word split on a tiny word list -----------------------------------------


def test_split_prefers_more_coverage() -> None:
    words = frozenset({"abc", "bcdef"})
    # "abc" covers 3 letters, "bcdef" covers 5 -> pick bcdef
    assert split_into_words("abcdef", words, min_len=3, max_len=5) == (["bcdef"], 5)


def test_split_prefers_fewer_words_on_tie() -> None:
    words = frozenset({"cat", "dog", "catdog"})
    assert split_into_words("catdog", words, min_len=3, max_len=6) == (["catdog"], 6)


def test_split_treats_hyphens_and_digits_as_boundaries() -> None:
    words = frozenset({"cat", "dog", "tdo"})
    # "tdo" would span the hyphen; it must not be used
    assert split_into_words("cat-dog", words, min_len=3, max_len=3) == (["cat", "dog"], 3)
    assert split_into_words("cat9dog", words, min_len=3, max_len=3) == (["cat", "dog"], 3)


def test_split_ignores_words_shorter_than_min_len() -> None:
    words = frozenset({"ab", "cat"})
    assert split_into_words("abcat", words, min_len=3, max_len=3) == (["cat"], 3)


# --- dictionary features on the committed word list -------------------------


@pytest.mark.parametrize(
    ("domain", "count"),
    [
        ("subjectclean.net", 2),  # suppobox_1
        ("weakgrow.net", 2),  # suppobox_2
        ("constanceboniface.net", 2),  # suppobox_3 (first names)
        ("board-care-bedroom.com", 3),  # matsnu
    ],
)
def test_word_based_dga_is_fully_covered(domain: str, count: int) -> None:
    f = feats(domain)
    assert f["dict_coverage"] == 1.0
    assert f["dict_word_count"] == count
    assert f["is_dict_word"] == 0.0


def test_random_dga_has_no_dictionary_coverage() -> None:
    f = feats("qxxsoxcwrcplo.net")  # cryptolocker
    assert f["dict_coverage"] == 0.0
    assert f["dict_word_count"] == 0
    assert f["longest_dict_word_ratio"] == 0.0


def test_whole_word_label() -> None:
    f = feats("google.com")
    assert f["is_dict_word"] == 1.0
    assert f["longest_dict_word_ratio"] == 1.0
    assert f["dict_word_count"] == 1


def test_load_wordlist_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "words.txt"
    path.write_text("cat\n\n dog \n", encoding="utf-8")
    assert features.load_wordlist(path) == frozenset({"cat", "dog"})
