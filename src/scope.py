"""Is a domain inside what the model can judge? Flags private suffixes (C1) and punycode (C2).

The features (src.features) look at the second-level label under the *public* suffix list. Under a
private suffix such as blogspot.com or duckdns.org, that label is the provider's name ("blogspot"),
so the generated part (abcxyz in abcxyz.blogspot.com) is never seen. Punycode (xn--) labels encode
internationalized names, which the ASCII training data does not cover.

Such domains still get a prediction; the API marks them in_scope: false with a warning. Detection
uses a second, offline tldextract extractor that includes the PSL's private section; features.py is
unchanged.
"""

import tldextract

from src.features import second_level_label

PUNYCODE_PREFIX = "xn--"

_PRIVATE_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)


def private_suffix(domain: str) -> str | None:
    """The private suffix a domain sits under (abc.blogspot.com -> 'blogspot.com'), else None.

    The provider's own name (blogspot.com) has no label below the private suffix: in scope.
    """
    result = _PRIVATE_EXTRACTOR(domain)
    if result.is_private and result.domain:
        return result.suffix
    return None


def has_punycode(domain: str) -> bool:
    return any(label.startswith(PUNYCODE_PREFIX) for label in domain.split("."))


def check_scope(domain: str) -> tuple[bool, str | None]:
    """(in_scope, warning) for a normalized domain; warning is None when in scope."""
    warnings: list[str] = []
    suffix = private_suffix(domain)
    if suffix is not None:
        below = domain.removesuffix("." + suffix).split(".")[-1]
        warnings.append(
            f"'{domain}' is under the private suffix '{suffix}' (a hosting or dynamic-DNS "
            f"provider); the model only sees the label '{second_level_label(domain)}', so the "
            f"prediction says nothing about '{below}'."
        )
    if has_punycode(domain):
        warnings.append(
            f"'{domain}' contains a punycode ({PUNYCODE_PREFIX}) label; the model was trained on "
            "ASCII names and its features do not describe internationalized domains."
        )
    return (not warnings, " ".join(warnings) or None)
