"""Data preparation: validate data/sample/, build train / test_seen / test_future parquet splits.

Steps:
1. Load the committed sample (UMUDGA DGA domains + Tranco benign domains).
2. Validate, printing how many rows each check drops.
3. Add the second-level label (sld) and one column per FEATURE_NAMES entry (from src.features).
4. Split (CLAUDE.md design decisions 4-6):
   - held-out families go only to test_future
   - training families + benign: GroupShuffleSplit 80/20 grouped by sld -> train / test_seen
   - test_future = held-out DGA rows + the benign rows of test_seen
5. Write the three parquet files and split_report.json.

Run with: python -m src.prepare_data
"""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config
from src.features import FEATURE_NAMES, extract_features, normalize_domain, second_level_label

COLUMNS = ["domain", "sld", "family", "label", *FEATURE_NAMES]
VALID_DOMAIN = re.compile(r"^[a-z0-9.-]+$")
MAX_LABEL_LEN = 63  # DNS limit for one label
MAX_DOMAIN_LEN = 253  # DNS limit for a full name
TEST_SIZE = 0.2

UMUDGA_SAMPLE_FILE = config.UMUDGA_SAMPLE_PATH.name
TRANCO_SAMPLE_FILE = config.TRANCO_SAMPLE_PATH.name


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_sample(sample_dir: Path) -> pd.DataFrame:
    """Load both sample files into one frame with columns domain, family, label."""
    dga = pd.read_csv(sample_dir / UMUDGA_SAMPLE_FILE, dtype=str)
    dga["label"] = "dga"
    benign = pd.read_csv(sample_dir / TRANCO_SAMPLE_FILE, dtype=str)
    benign["family"] = "benign"
    benign["label"] = "benign"
    columns = ["domain", "family", "label"]
    return pd.concat([dga[columns], benign[columns]], ignore_index=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _drop(df: pd.DataFrame, mask: pd.Series, name: str, drops: dict[str, int]) -> pd.DataFrame:
    """Drop rows where mask is True, record and print the count."""
    drops[name] = int(mask.sum())
    print(f"  {name}: dropped {drops[name]}")
    return df[~mask]


def _featurize(domain: str) -> tuple[str, list[float]] | None:
    """(sld, features) for a domain, or None if it has no second-level label."""
    try:
        features = extract_features(domain)
    except ValueError:
        return None
    return second_level_label(domain), features


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean the raw sample and add sld + feature columns. Returns (clean frame, drop counts)."""
    drops: dict[str, int] = {}
    print("Validation:")

    df = _drop(df, df["family"].isin(config.EXCLUDED_FAMILIES), "excluded_family", drops)

    empty = df["domain"].isna() | df["family"].isna()
    empty |= df["domain"].fillna("").str.strip() == ""
    df = _drop(df, empty, "null_or_empty", drops)

    df = df.assign(domain=df["domain"].map(normalize_domain))
    invalid = ~df["domain"].map(lambda d: bool(VALID_DOMAIN.match(d)))
    df = _drop(df, invalid, "invalid_characters", drops)

    results = df["domain"].map(_featurize)
    df = _drop(df, results.isna(), "empty_second_level_label", drops)
    results = results[results.notna()]
    df = df.assign(sld=[sld for sld, _ in results])
    features = pd.DataFrame([f for _, f in results], columns=FEATURE_NAMES, index=df.index)
    df = pd.concat([df, features], axis=1)

    too_long = (df["sld"].str.len() > MAX_LABEL_LEN) | (df["domain"].str.len() > MAX_DOMAIN_LEN)
    df = _drop(df, too_long, "length_out_of_range", drops)

    # A domain labelled both benign and dga is ambiguous: drop every copy of it.
    labels_per_domain = df.groupby("domain")["label"].transform("nunique")
    df = _drop(df, labels_per_domain > 1, "on_both_sides", drops)

    df = _drop(df, df["domain"].duplicated(keep="first"), "duplicate_domain", drops)

    return df[COLUMNS].reset_index(drop=True), drops


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {"train", "test_seen", "test_future"} following CLAUDE.md decisions 4-6."""
    is_held_out = df["family"].isin(config.HELD_OUT_FAMILIES)
    held_out, pool = df[is_held_out], df[~is_held_out]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=config.SEED)
    train_idx, test_idx = next(splitter.split(pool, groups=pool["sld"]))
    train, test_seen = pool.iloc[train_idx], pool.iloc[test_idx]

    test_future = pd.concat([held_out, test_seen[test_seen["label"] == "benign"]])
    return {
        "train": train.reset_index(drop=True),
        "test_seen": test_seen.reset_index(drop=True),
        "test_future": test_future.reset_index(drop=True),
    }


def split_report(splits: dict[str, pd.DataFrame], drops: dict[str, int]) -> dict:
    """Counts per split and per family, plus the leakage check between train and test_seen."""
    shared = set(splits["train"]["sld"]) & set(splits["test_seen"]["sld"])
    report = {
        "validation_drops": drops,
        "totals": {},
        "rows_per_family": {},
        "shared_sld_train_test_seen": len(shared),
        "shared_sld_train_test_future": len(
            set(splits["train"]["sld"]) & set(splits["test_future"]["sld"])
        ),
    }
    for name, frame in splits.items():
        counts = frame["label"].value_counts()
        report["totals"][name] = {
            "rows": len(frame),
            "benign": int(counts.get("benign", 0)),
            "dga": int(counts.get("dga", 0)),
            "benign_fraction": round(float(counts.get("benign", 0)) / max(len(frame), 1), 4),
        }
        per_family = frame["family"].value_counts().sort_index()
        report["rows_per_family"][name] = {fam: int(n) for fam, n in per_family.items()}
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def prepare_data(sample_dir: Path, out_dir: Path) -> dict:
    """Validate the sample, split it, write parquet files + split_report.json; return the report."""
    df, drops = validate(load_sample(sample_dir))
    splits = split(df)
    report = split_report(splits, drops)
    if report["shared_sld_train_test_seen"] != 0:
        raise RuntimeError("Leakage: some sld values are in both train and test_seen")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / config.SPLIT_REPORT_PATH.name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = prepare_data(config.SAMPLE_DIR, config.PROCESSED_DIR)
    for name, totals in report["totals"].items():
        print(f"{name}: {totals}")
    print(f"shared sld train/test_seen: {report['shared_sld_train_test_seen']}")


if __name__ == "__main__":
    main()
