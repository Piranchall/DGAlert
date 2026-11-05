"""Tests for the hyphen helpers in scripts/hyphen_evasion.py (no model or processed data needed)."""

import pytest

from scripts.hyphen_evasion import hyphenated_domain, insert_hyphen
from src.features import second_level_label


@pytest.mark.parametrize(
    ("label", "expected"),
    [("abcdef", "abc-def"), ("abcde", "ab-cde"), ("ab", "a-b")],
)
def test_insert_hyphen_in_the_middle(label: str, expected: str) -> None:
    assert insert_hyphen(label) == expected


@pytest.mark.parametrize(
    ("sld", "domain", "expected"),
    [
        ("moiqidfvjjhjmndqp", "moiqidfvjjhjmndqp.in", "moiqidfv-jjhjmndqp.in"),
        ("teco", "teco.com.tw", "te-co.com.tw"),
    ],
)
def test_hyphenated_domain_keeps_the_suffix(sld: str, domain: str, expected: str) -> None:
    result = hyphenated_domain(sld, domain)
    assert result == expected
    assert second_level_label(result) == insert_hyphen(sld)
