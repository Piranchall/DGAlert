"""Build the committed sample in data/sample/ from data/raw/ (run once, commit the output).

UMUDGA: for every family folder not in EXCLUDED_FAMILIES, read list/10000.txt only, normalize and
deduplicate, then sample up to SAMPLE_PER_FAMILY domains (seed 42). Held-out families are sampled
too; prepare_data decides where their rows go.

Tranco: drop _wildcard_ rows, punycode (xn--) domains and bare public suffixes, then sample the same
number of benign domains as there are sampled DGA rows from training families, spread evenly over
the ten 100k-rank bands (not just the top of the list).

Domains found on both sides (Tranco and a DGA list) are dropped from both before sampling.

Run with: python -m scripts.make_sample
"""

import hashlib
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from src import config
from src.features import normalize_domain, second_level_label

RANK_BAND_SIZE = 100_000
N_RANK_BANDS = 10
GZIP = {"method": "gzip", "mtime": 0}  # mtime=0 so the same data gives the same bytes


# ---------------------------------------------------------------------------
# Reading and cleaning
# ---------------------------------------------------------------------------


def read_family(family_dir: Path) -> list[str]:
    """Normalized, de-duplicated domains from one family's list/10000.txt (file order kept)."""
    lines = (family_dir / "list" / "10000.txt").read_text(encoding="utf-8").splitlines()
    domains = (normalize_domain(line) for line in lines)
    return list(dict.fromkeys(d for d in domains if d))


def read_umudga(umudga_dir: Path, excluded: list[str]) -> tuple[dict[str, list[str]], int]:
    """Read every non-excluded family. Returns ({family: domains}, raw line count)."""
    families = {}
    raw_lines = 0
    for family_dir in sorted(p for p in umudga_dir.iterdir() if p.is_dir()):
        if family_dir.name in excluded:
            continue
        lines = (family_dir / "list" / "10000.txt").read_text(encoding="utf-8").splitlines()
        raw_lines += len(lines)
        families[family_dir.name] = read_family(family_dir)
    return families, raw_lines


def drop_cross_family(families: dict[str, list[str]]) -> tuple[dict[str, list[str]], int]:
    """Drop domains that appear in more than one family (their family would be ambiguous)."""
    counts = Counter(d for domains in families.values() for d in domains)
    shared = {d for d, n in counts.items() if n > 1}
    cleaned = {fam: [d for d in domains if d not in shared] for fam, domains in families.items()}
    return cleaned, len(shared)


def read_tranco(path: Path) -> pd.DataFrame:
    """Read the Tranco CSV (rank,domain; no header; CRLF handled by pandas)."""
    df = pd.read_csv(path, header=None, names=["tranco_rank", "domain"], dtype={"domain": str})
    df["domain"] = df["domain"].map(normalize_domain)
    return df


def clean_tranco(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop _wildcard_ rows, punycode domains and bare public suffixes; return counts dropped."""
    drops = {}
    wildcard = df["domain"].str.contains("_wildcard_", regex=False)
    drops["wildcard"] = int(wildcard.sum())
    df = df[~wildcard]

    punycode = df["domain"].str.contains("xn--", regex=False)
    drops["punycode"] = int(punycode.sum())
    df = df[~punycode]

    bare_suffix = df["domain"].map(second_level_label) == ""
    drops["bare_public_suffix"] = int(bare_suffix.sum())
    df = df[~bare_suffix]
    return df.reset_index(drop=True), drops


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_families(families: dict[str, list[str]], n: int, seed: int) -> pd.DataFrame:
    """Up to n domains per family, sampled with a fixed seed. Columns: domain, family."""
    parts = []
    for family, domains in sorted(families.items()):
        frame = pd.DataFrame({"domain": domains, "family": family})
        parts.append(frame.sample(n=min(n, len(frame)), random_state=seed))
    return pd.concat(parts).sort_values(["family", "domain"]).reset_index(drop=True)


def sample_benign(tranco: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n domains spread evenly over the ten 100k-rank bands. Columns: domain, tranco_rank."""
    band = (tranco["tranco_rank"] - 1) // RANK_BAND_SIZE
    per_band = [n // N_RANK_BANDS + (1 if i < n % N_RANK_BANDS else 0) for i in range(N_RANK_BANDS)]
    parts = [
        tranco[band == i].sample(n=per_band[i], random_state=seed) for i in range(N_RANK_BANDS)
    ]
    sample = pd.concat(parts)[["domain", "tranco_rank"]]
    return sample.sort_values("tranco_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# SAMPLE_INFO.md
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sample_info(path: Path, stats: dict, dga: pd.DataFrame, benign: pd.DataFrame) -> None:
    """Write the human-readable record of how the sample was made."""
    held = set(config.HELD_OUT_FAMILIES)
    per_family = dga["family"].value_counts().sort_index()
    family_rows = "\n".join(
        f"| {fam} | {'held-out' if fam in held else 'train'} | "
        f"{stats['family_available'][fam]} | {count} |"
        for fam, count in per_family.items()
    )
    band = (benign["tranco_rank"] - 1) // RANK_BAND_SIZE
    band_rows = "\n".join(
        f"| {i * RANK_BAND_SIZE + 1:,}-{(i + 1) * RANK_BAND_SIZE:,} | {int((band == i).sum())} |"
        for i in range(N_RANK_BANDS)
    )
    drops = stats["tranco_drops"]
    text = f"""# Sample info

Generated by `python -m scripts.make_sample` on {date.today().isoformat()}.
Do not edit by hand: rerun the script instead.

## Sources
- **UMUDGA** (Zago, Gil Pérez, Martínez Pérez, 2020, "UMUDGA: A dataset for profiling
  algorithmically generated domain names in botnet detection", Data in Brief). Only
  `<family>/list/10000.txt` is read; the family name is the folder name.
- **Tranco** list `{config.TRANCO_LIST_ID}`: https://tranco-list.eu/list/{config.TRANCO_LIST_ID}/1000000
  - file `data/raw/tranco/top-1m.csv`, SHA-256 `{stats["tranco_sha256"]}`
  - verified on 2026-11-05 to be byte-identical to a fresh download of list
    {config.TRANCO_LIST_ID}

## Parameters
- Domains per UMUDGA family: up to **{stats["n_per_family"]}**
- Seed: **{config.SEED}**
- Benign rows = sampled DGA rows from training families = **{len(benign)}**, spread evenly over
  the ten 100k-rank bands of Tranco (not the top N)

## Excluded families (never used)
{", ".join(config.EXCLUDED_FAMILIES)}

- corebot, symmi, kraken_v1, kraken_v2: the generated string sits in a dynamic-DNS subdomain
  (e.g. `x.ddns.net`, `x.mooo.com`), so the second-level label is just the provider name.
  Out of scope for a second-level-label classifier.
- legit: UMUDGA's own benign list. It overlaps Tranco heavily, and Tranco is our benign source.

## Held-out families (only in test_future)
{", ".join(config.HELD_OUT_FAMILIES)}

Chosen by the team ({len(config.HELD_OUT_FAMILIES)} of {stats["n_families_used"]} usable
families, about 30%) to simulate future malware the model has never seen. The set mixes
word-based (matsnu, nymaim), pronounceable or seed-based (banjori, simda, sisron) and
random-character families.

## Drop counts
| Step | Dropped |
|---|---|
| UMUDGA: empty or duplicate lines in a family (normalized) | {stats["umudga_dedup_dropped"]} |
| UMUDGA: domains appearing in more than one family | {stats["cross_family_dropped"]} |
| Tranco: `_wildcard_` rows | {drops["wildcard"]} |
| Tranco: punycode (`xn--`) domains | {drops["punycode"]} |
| Tranco: bare public suffixes (empty second-level label) | {drops["bare_public_suffix"]} |
| Domains in both Tranco and a DGA list (dropped from both sides) | {stats["cross_side_dropped"]} |

## Rows per UMUDGA family
| Family | Role | Available after cleaning | Sampled |
|---|---|---|---|
{family_rows}

Total DGA rows: {len(dga)} ({stats["dga_train_rows"]} from training families,
{len(dga) - stats["dga_train_rows"]} from held-out families).

## Benign rows per Tranco rank band
| Rank band | Rows |
|---|---|
{band_rows}

Total benign rows: {len(benign)}
"""
    path.write_text(text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_sample(n_per_family: int = config.SAMPLE_PER_FAMILY, seed: int = config.SEED) -> dict:
    """Build and write the sample files. Returns the stats written to SAMPLE_INFO.md."""
    family_dirs = {p.name for p in config.UMUDGA_DIR.iterdir() if p.is_dir()}
    unknown = set(config.HELD_OUT_FAMILIES + config.EXCLUDED_FAMILIES) - family_dirs
    if unknown:
        raise ValueError(f"Families in config.py not found in {config.UMUDGA_DIR}: {unknown}")

    families, raw_lines = read_umudga(config.UMUDGA_DIR, config.EXCLUDED_FAMILIES)
    after_dedup = sum(len(d) for d in families.values())
    families, cross_family = drop_cross_family(families)

    tranco, tranco_drops = clean_tranco(read_tranco(config.TRANCO_PATH))

    # Drop domains present on both sides, from both sides.
    dga_domains = {d for domains in families.values() for d in domains}
    both = dga_domains & set(tranco["domain"])
    families = {fam: [d for d in domains if d not in both] for fam, domains in families.items()}
    tranco = tranco[~tranco["domain"].isin(both)]

    dga = sample_families(families, n_per_family, seed)
    dga_train_rows = int((~dga["family"].isin(config.HELD_OUT_FAMILIES)).sum())
    benign = sample_benign(tranco, dga_train_rows, seed)

    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    dga.to_csv(config.UMUDGA_SAMPLE_PATH, index=False, compression=GZIP, lineterminator="\n")
    benign.to_csv(config.TRANCO_SAMPLE_PATH, index=False, compression=GZIP, lineterminator="\n")

    stats = {
        "n_per_family": n_per_family,
        "n_families_used": len(families),
        "umudga_raw_lines": raw_lines,
        "umudga_dedup_dropped": raw_lines - after_dedup,
        "cross_family_dropped": cross_family,
        "tranco_drops": tranco_drops,
        "cross_side_dropped": len(both),
        "family_available": {fam: len(domains) for fam, domains in families.items()},
        "dga_rows": len(dga),
        "dga_train_rows": dga_train_rows,
        "benign_rows": len(benign),
        "tranco_sha256": sha256_of(config.TRANCO_PATH),
    }
    write_sample_info(config.SAMPLE_INFO_PATH, stats, dga, benign)
    return stats


def main() -> None:
    stats = make_sample()
    for key, value in stats.items():
        if key != "family_available":
            print(f"{key}: {value}")
    sizes = {p.name: p.stat().st_size for p in config.SAMPLE_DIR.iterdir() if p.is_file()}
    print(f"sample files (bytes): {sizes}, total {sum(sizes.values()) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
