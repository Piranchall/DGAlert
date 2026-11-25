"""Retraining data: frozen eval set, phase-in pool and validation split, from the M1 splits.

Reads the three parquet files written by src.prepare_data (which this module does not change) and
derives:
- frozen eval slices: about 20% of each held-out family, chosen by a hash of the second-level
  label (sld); the other ~80% of held-out rows form the phase-in pool;
- the frozen eval set = test_seen + all frozen eval slices; its content hash and row counts per
  family go to the committed reports/eval_frozen.json;
- a validation split: about 10% of train, chosen by the same hash rule with another salt; the rest
  of train (train_fit) is what retraining fits on.

No sld is shared between train_fit, validation, phase_in_pool and eval_frozen (checked; a shared
sld raises).

Run with: python -m src.prepare_retrain (after python -m src.prepare_data)
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src import config

PARTS = ("train_fit", "validation", "phase_in_pool", "eval_frozen")
HASH_INPUT = "SHA-256 of the sorted lines '<domain>,<label>\\n' (UTF-8), one per row"


# ---------------------------------------------------------------------------
# Hash rules
# ---------------------------------------------------------------------------


def sld_fraction(sld: str, salt: str) -> float:
    """A number in [0, 1) fixed by the sld and the salt: sha256("<salt>:<sld>") / 2**64."""
    digest = hashlib.sha256(f"{salt}:{sld}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def selected(frame: pd.DataFrame, salt: str, fraction: float) -> pd.Series:
    """True for the rows whose sld falls in the first `fraction` of the hash range."""
    return frame["sld"].map(lambda sld: sld_fraction(sld, salt) < fraction).astype(bool)


def content_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over the sorted '<domain>,<label>' lines, independent of row order."""
    pairs = zip(frame["domain"], frame["label"], strict=True)
    lines = sorted(f"{domain},{label}\n" for domain, label in pairs)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted by domain, so the output does not depend on the input order."""
    return frame.sort_values("domain", kind="mergesort").reset_index(drop=True)


def shared_sld_counts(parts: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Number of sld values shared by each pair of parts (all must be 0)."""
    slds = {name: set(frame["sld"]) for name, frame in parts.items()}
    return {f"{a}/{b}": len(slds[a] & slds[b]) for a, b in combinations(parts, 2)}


def build(
    train: pd.DataFrame, test_seen: pd.DataFrame, test_future: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Derive train_fit, validation, phase_in_pool and eval_frozen from the M1 splits."""
    held_out = test_future[
        (test_future["label"] == "dga") & test_future["family"].isin(config.HELD_OUT_FAMILIES)
    ]
    families_per_sld = held_out.groupby("sld")["family"].nunique()
    if (families_per_sld > 1).any():
        examples = sorted(families_per_sld[families_per_sld > 1].index)[:5]
        raise RuntimeError(f"sld shared by several held-out families: {examples}")

    frozen = selected(held_out, config.FROZEN_EVAL_SALT, config.FROZEN_EVAL_FRACTION)
    in_validation = selected(train, config.VALIDATION_SALT, config.VALIDATION_FRACTION)
    parts = {
        "train_fit": _sorted(train[~in_validation]),
        "validation": _sorted(train[in_validation]),
        "phase_in_pool": _sorted(held_out[~frozen]),
        "eval_frozen": _sorted(pd.concat([test_seen, held_out[frozen]])),
    }
    shared = {pair: n for pair, n in shared_sld_counts(parts).items() if n}
    if shared:
        raise RuntimeError(f"Leakage: sld values shared between parts: {shared}")
    return parts


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _family_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {family: int(n) for family, n in frame["family"].value_counts().sort_index().items()}


def eval_frozen_report(eval_frozen: pd.DataFrame, n_test_seen: int) -> dict:
    """The committed description of the frozen eval set (deterministic: no times or paths)."""
    labels = eval_frozen["label"].value_counts()
    return {
        "sha256": content_hash(eval_frozen),
        "hash_input": HASH_INPUT,
        "rows": len(eval_frozen),
        "benign": int(labels.get("benign", 0)),
        "dga": int(labels.get("dga", 0)),
        "sources": {"test_seen": n_test_seen, "frozen_eval_slices": len(eval_frozen) - n_test_seen},
        "frozen_eval_slice_rule": {
            "families": sorted(config.HELD_OUT_FAMILIES),
            "salt": config.FROZEN_EVAL_SALT,
            "fraction": config.FROZEN_EVAL_FRACTION,
        },
        "rows_per_family": _family_counts(eval_frozen),
    }


def split_report(parts: dict[str, pd.DataFrame], train: pd.DataFrame, held_out_rows: int) -> dict:
    """Counts per part and family, the actual fractions per family, and the sld overlap checks."""
    validation_share = (
        parts["validation"]["family"].value_counts() / train["family"].value_counts()
    ).fillna(0.0)
    pool = parts["phase_in_pool"]["family"].value_counts()
    eval_frozen = parts["eval_frozen"]
    frozen_slices = eval_frozen[eval_frozen["family"].isin(config.HELD_OUT_FAMILIES)]
    frozen = frozen_slices["family"].value_counts()
    frozen_share = (frozen / (frozen.add(pool, fill_value=0))).fillna(0.0)
    return {
        "rows": {name: len(frame) for name, frame in parts.items()},
        "held_out_dga_rows": held_out_rows,
        "rows_per_family": {name: _family_counts(frame) for name, frame in parts.items()},
        "validation_fraction_per_family": {
            family: round(float(share), 4)
            for family, share in validation_share.sort_index().items()
        },
        "frozen_fraction_per_family": {
            family: round(float(share), 4) for family, share in frozen_share.sort_index().items()
        },
        "shared_sld": shared_sld_counts(parts),
        "phase_in_batches": config.PHASE_IN_BATCHES,
        "phase_in_control_families": config.PHASE_IN_CONTROL_FAMILIES,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def prepare_retrain(processed_dir: Path, out_dir: Path, eval_report_path: Path) -> dict:
    """Read the M1 splits from processed_dir, write the four parts and both reports.

    Returns the committed eval_frozen report.
    """
    splits = {
        name: pd.read_parquet(processed_dir / path.name)
        for name, path in (
            ("train", config.TRAIN_PATH),
            ("test_seen", config.TEST_SEEN_PATH),
            ("test_future", config.TEST_FUTURE_PATH),
        )
    }
    parts = build(splits["train"], splits["test_seen"], splits["test_future"])
    held_out_rows = len(parts["phase_in_pool"]) + (
        len(parts["eval_frozen"]) - len(splits["test_seen"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_fit": config.TRAIN_FIT_PATH,
        "validation": config.VALIDATION_PATH,
        "phase_in_pool": config.PHASE_IN_POOL_PATH,
        "eval_frozen": config.EVAL_FROZEN_PATH,
    }
    for name, frame in parts.items():
        frame.to_parquet(out_dir / paths[name].name, index=False)
    report = split_report(parts, splits["train"], held_out_rows)
    (out_dir / config.RETRAIN_SPLIT_REPORT_PATH.name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    eval_report = eval_frozen_report(parts["eval_frozen"], len(splits["test_seen"]))
    eval_report_path.parent.mkdir(parents=True, exist_ok=True)
    eval_report_path.write_text(json.dumps(eval_report, indent=2) + "\n", encoding="utf-8")
    return eval_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=config.EVAL_FROZEN_REPORT_PATH,
        help="where to write the frozen eval report (default: the committed "
        "reports/eval_frozen.json). The retrain workflow writes it elsewhere, so the committed "
        "hash stays the reference the gate checks against.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = prepare_retrain(config.PROCESSED_DIR, config.PROCESSED_DIR, args.eval_report)
    split = json.loads(config.RETRAIN_SPLIT_REPORT_PATH.read_text(encoding="utf-8"))
    for name, rows in split["rows"].items():
        print(f"{name}: {rows} rows")
    print(f"shared sld: {split['shared_sld']}")
    print(f"eval_frozen sha256: {report['sha256']} ({report['rows']} rows)")


if __name__ == "__main__":
    main()
