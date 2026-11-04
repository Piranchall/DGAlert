"""Build data/reference/english_words.txt from SCOWL (run once, commit the output).

Source: SCOWL (Spell Checker Oriented Word Lists) by Kevin Atkinson, http://wordlist.aspell.net/
Version 2020.12.07, size 50, American English. The licence is copied verbatim to
data/reference/SCOWL-LICENSE.txt (it must stay next to the word list).

What we keep:
- categories: words, upper (capitalised words such as first names) and proper-names,
  for both "english" and "american" spellings, sizes 10-50 (what SCOWL's en_US size-50 list holds,
  minus abbreviations and contractions)
- entries lowercased; only plain a-z words of at least MIN_WORD_LEN letters
  (drops possessives like "aaron's", accented words, and 1-2 letter words that match everywhere)

Run with: python -m scripts.build_wordlist
"""

import hashlib
import re
import tarfile
import urllib.request
from pathlib import Path

from src.config import RAW_DIR, WORDLIST_LICENSE_PATH, WORDLIST_PATH

SCOWL_VERSION = "2020.12.07"
SCOWL_URL = (
    "https://downloads.sourceforge.net/project/wordlist/SCOWL/"
    f"{SCOWL_VERSION}/scowl-{SCOWL_VERSION}.tar.gz"
)
SCOWL_SHA256 = "5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc"
TARBALL_PATH = RAW_DIR / "scowl" / f"scowl-{SCOWL_VERSION}.tar.gz"  # gitignored
DOWNLOAD_TIMEOUT_S = 60

SPELLINGS = ["english", "american"]
CATEGORIES = ["words", "upper", "proper-names"]
SIZES = [10, 20, 35, 40, 50]
MIN_WORD_LEN = 3
WORD_PATTERN = re.compile(r"^[a-z]+$")


def download_tarball(path: Path = TARBALL_PATH) -> Path:
    """Download the SCOWL tarball if it is not already present, then verify its checksum.

    The download goes to a .part file that is renamed only when complete, so an interrupted
    download is never mistaken for a finished one. If a mirror stalls, the timeout raises an error;
    just run the script again.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        print(f"Downloading {SCOWL_URL}")
        with urllib.request.urlopen(SCOWL_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:
            part.write_bytes(response.read())
        part.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SCOWL_SHA256:
        raise ValueError(f"Checksum mismatch for {path}: {digest}")
    return path


def clean_words(lines: list[str], min_len: int = MIN_WORD_LEN) -> set[str]:
    """Lowercase entries and keep only plain a-z words of at least min_len letters."""
    words = set()
    for line in lines:
        word = line.strip().lower()
        if len(word) >= min_len and WORD_PATTERN.match(word):
            words.add(word)
    return words


def read_scowl_words(tarball: Path) -> set[str]:
    """Read the selected SCOWL list files straight from the tarball (no extraction)."""
    prefix = f"scowl-{SCOWL_VERSION}/final"
    wanted = {
        f"{prefix}/{spelling}-{category}.{size}"
        for spelling in SPELLINGS
        for category in CATEGORIES
        for size in SIZES
    }
    words: set[str] = set()
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name in wanted:
                raw = tar.extractfile(member).read()
                words |= clean_words(raw.decode("iso-8859-1").splitlines())
    return words


def read_scowl_license(tarball: Path) -> str:
    """Return SCOWL's Copyright file text, unchanged."""
    with tarfile.open(tarball, "r:gz") as tar:
        member = tar.getmember(f"scowl-{SCOWL_VERSION}/Copyright")
        return tar.extractfile(member).read().decode("iso-8859-1")


def main() -> None:
    tarball = download_tarball()
    words = sorted(read_scowl_words(tarball))
    WORDLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORDLIST_PATH.write_text("\n".join(words) + "\n", encoding="utf-8", newline="\n")
    WORDLIST_LICENSE_PATH.write_text(read_scowl_license(tarball), encoding="utf-8", newline="\n")
    print(f"Wrote {len(words)} words to {WORDLIST_PATH}")


if __name__ == "__main__":
    main()
