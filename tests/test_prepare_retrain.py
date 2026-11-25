"""Tests for src.prepare_retrain: frozen eval set, phase-in pool, validation split and their hash.

Synthetic frames for the rules; one integration test runs prepare_data + prepare_retrain on the
committed data/sample (never data/raw) and compares with the committed reports/eval_frozen.json.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.prepare_data import prepare_data
from src.prepare_retrain import (
    HASH_INPUT,
    PARTS,
    build,
    content_hash,
    eval_frozen_report,
    prepare_retrain,
    shared_sld_counts,
    sld_fraction,
)

ROWS_PER_FAMILY = 2000
TRAIN_FAMILIES = ["fam_a", "fam_b", "benign"]
HELD_FAMILIES = ["matsnu", "nymaim"]  # real held-out families, so config needs no patching
M1_ROWS = {"train": 153627, "test_seen": 38373, "test_future": 61141}


def name(family: str, i: int) -> str:
    """A deterministic, unique second-level label for row i of a family."""
    return family.replace("_", "") + hashlib.sha256(f"{family}{i}".encode()).hexdigest()[:10]


def frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(domain, family, label) rows -> frame with the sld column, like the M1 splits."""
    df = pd.DataFrame(rows, columns=["domain", "family", "label"])
    return df.assign(sld=df["domain"].str.split(".").str[0])


def label_of(family: str) -> str:
    return "benign" if family == "benign" else "dga"


@pytest.fixture
def splits() -> dict[str, pd.DataFrame]:
    """Synthetic train / test_seen / test_future with ROWS_PER_FAMILY rows per family."""
    train = frame(
        [
            (f"{name(fam, i)}.com", fam, label_of(fam))
            for fam in TRAIN_FAMILIES
            for i in range(ROWS_PER_FAMILY)
        ]
    )
    test_seen = frame(
        [
            (f"{name(fam, -i - 1)}.net", fam, label_of(fam))
            for fam in TRAIN_FAMILIES
            for i in range(300)
        ]
    )
    held = frame(
        [
            (f"{name(fam, i)}.org", fam, "dga")
            for fam in HELD_FAMILIES
            for i in range(ROWS_PER_FAMILY)
        ]
    )
    test_future = pd.concat([held, test_seen[test_seen["label"] == "benign"]], ignore_index=True)
    return {"train": train, "test_seen": test_seen, "test_future": test_future}


def parts_of(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return build(splits["train"], splits["test_seen"], splits["test_future"])


# ---------------------------------------------------------------------------
# Hash rules
# ---------------------------------------------------------------------------


def test_sld_fraction_is_fixed_by_sld_and_salt() -> None:
    value = sld_fraction("example", "frozen-eval")
    assert value == sld_fraction("example", "frozen-eval")
    assert 0.0 <= value < 1.0
    assert value != sld_fraction("example", "validation")
    digest = hashlib.sha256(b"frozen-eval:example").digest()
    assert value == int.from_bytes(digest[:8], "big") / 2**64


def test_content_hash_format_and_order_independence() -> None:
    df = pd.DataFrame({"domain": ["b.com", "a.com"], "label": ["dga", "benign"]})
    expected = hashlib.sha256(b"a.com,benign\nb.com,dga\n").hexdigest()
    assert content_hash(df) == expected
    assert content_hash(df.iloc[::-1]) == expected
    assert content_hash(df.assign(label=["benign", "benign"])) != expected
    assert "sorted" in HASH_INPUT and "<domain>,<label>" in HASH_INPUT


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def test_parts_cover_the_inputs(splits: dict[str, pd.DataFrame]) -> None:
    parts = parts_of(splits)
    assert tuple(parts) == PARTS
    assert len(parts["train_fit"]) + len(parts["validation"]) == len(splits["train"])
    held = splits["test_future"][splits["test_future"]["label"] == "dga"]
    frozen_slices = len(parts["eval_frozen"]) - len(splits["test_seen"])
    assert len(parts["phase_in_pool"]) + frozen_slices == len(held)
    # Every test_seen row is in the frozen eval set; the pool holds only held-out DGA rows.
    assert set(splits["test_seen"]["domain"]) <= set(parts["eval_frozen"]["domain"])
    assert set(parts["phase_in_pool"]["family"]) == set(HELD_FAMILIES)
    assert set(parts["phase_in_pool"]["label"]) == {"dga"}


def test_no_sld_shared_between_parts(splits: dict[str, pd.DataFrame]) -> None:
    counts = shared_sld_counts(parts_of(splits))
    assert len(counts) == 6
    assert set(counts.values()) == {0}


@pytest.mark.parametrize(
    ("part", "families", "low", "high"),
    [
        ("validation", TRAIN_FAMILIES, 0.08, 0.12),  # about 10% of each family of train
        ("frozen", HELD_FAMILIES, 0.17, 0.23),  # about 20% of each held-out family
    ],
)
def test_fractions_per_family(
    splits: dict[str, pd.DataFrame], part: str, families: list[str], low: float, high: float
) -> None:
    parts = parts_of(splits)
    rows = parts["validation"] if part == "validation" else parts["eval_frozen"]
    for family in families:
        share = (rows["family"] == family).sum() / ROWS_PER_FAMILY
        assert low <= share <= high, (part, family, share)


def test_output_does_not_depend_on_input_order(splits: dict[str, pd.DataFrame]) -> None:
    shuffled = {k: v.sample(frac=1.0, random_state=1) for k, v in splits.items()}
    first, second = parts_of(splits), parts_of(shuffled)
    for part in PARTS:
        pd.testing.assert_frame_equal(first[part], second[part])


def test_membership_is_stable_when_other_rows_change(splits: dict[str, pd.DataFrame]) -> None:
    """A row's side depends only on its sld: adding a family or rows changes no existing row."""
    before = parts_of(splits)
    extra_held = frame([(f"{name('simda', i)}.org", "simda", "dga") for i in range(500)])
    extra_train = frame([(f"{name('fam_c', i)}.com", "fam_c", "dga") for i in range(500)])
    grown = {
        "train": pd.concat([splits["train"], extra_train], ignore_index=True),
        "test_seen": splits["test_seen"],
        "test_future": pd.concat([splits["test_future"], extra_held], ignore_index=True),
    }
    after = parts_of(grown)
    for part in PARTS:
        old = set(before[part]["domain"])
        assert old <= set(after[part]["domain"]), part
    assert content_hash(after["eval_frozen"]) != content_hash(before["eval_frozen"])
    old_frozen = before["eval_frozen"]
    new_frozen = after["eval_frozen"][after["eval_frozen"]["family"] != "simda"]
    assert content_hash(new_frozen) == content_hash(old_frozen)


def test_rows_with_one_sld_stay_together(splits: dict[str, pd.DataFrame]) -> None:
    # The same sld under many TLDs: every copy lands on the same side.
    same = frame([(f"sharedlabel.{tld}", "fam_a", "dga") for tld in ("com", "net", "org", "io")])
    splits["train"] = pd.concat([splits["train"], same], ignore_index=True)
    parts = parts_of(splits)
    homes = [p for p in ("train_fit", "validation") if "sharedlabel" in set(parts[p]["sld"])]
    assert len(homes) == 1
    assert (parts[homes[0]]["sld"] == "sharedlabel").sum() == 4


def test_sld_in_two_held_out_families_raises(splits: dict[str, pd.DataFrame]) -> None:
    clash = frame([("clash.org", "matsnu", "dga"), ("clash.info", "nymaim", "dga")])
    splits["test_future"] = pd.concat([splits["test_future"], clash], ignore_index=True)
    with pytest.raises(RuntimeError, match="several held-out families"):
        parts_of(splits)


def test_sld_shared_between_parts_raises(splits: dict[str, pd.DataFrame]) -> None:
    # Invalid input (an M1 split with leakage) must not produce a leaking frozen eval set.
    leak = splits["train"].iloc[[0]].assign(domain=lambda d: d["sld"] + ".leak")
    splits["test_seen"] = pd.concat([splits["test_seen"], leak], ignore_index=True)
    with pytest.raises(RuntimeError, match="Leakage"):
        parts_of(splits)


def test_eval_frozen_report(splits: dict[str, pd.DataFrame]) -> None:
    parts = parts_of(splits)
    report = eval_frozen_report(parts["eval_frozen"], len(splits["test_seen"]))
    assert report == eval_frozen_report(parts["eval_frozen"], len(splits["test_seen"]))
    assert report["sha256"] == content_hash(parts["eval_frozen"])
    assert report["rows"] == len(parts["eval_frozen"]) == report["benign"] + report["dga"]
    assert sum(report["rows_per_family"].values()) == report["rows"]
    assert report["sources"]["test_seen"] == len(splits["test_seen"])
    assert report["frozen_eval_slice_rule"]["fraction"] == config.FROZEN_EVAL_FRACTION


# ---------------------------------------------------------------------------
# Phase-in batches
# ---------------------------------------------------------------------------


def test_phase_in_batches() -> None:
    batches = config.PHASE_IN_BATCHES
    assert batches[0] == ["matsnu"]
    phased = [family for batch in batches for family in batch]
    assert len(phased) == len(set(phased)), "a family is in two batches"
    assert set(phased) <= set(config.HELD_OUT_FAMILIES)
    control = config.PHASE_IN_CONTROL_FAMILIES
    assert set(control) == set(config.HELD_OUT_FAMILIES) - set(phased)
    assert not set(control) & set(phased)


def test_fractions_and_salts() -> None:
    assert config.FROZEN_EVAL_FRACTION == 0.2
    assert config.VALIDATION_FRACTION == 0.1
    assert config.FROZEN_EVAL_SALT != config.VALIDATION_SALT


# ---------------------------------------------------------------------------
# Integration: the committed sample
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """prepare_data + prepare_retrain on data/sample, written to a temp folder."""
    out = tmp_path_factory.mktemp("processed")
    m1 = prepare_data(config.SAMPLE_DIR, out)
    eval_report = prepare_retrain(out, out, out / "eval_frozen.json")
    split = json.loads((out / config.RETRAIN_SPLIT_REPORT_PATH.name).read_text(encoding="utf-8"))
    return {"m1": m1, "eval": eval_report, "split": split, "dir": out}


def test_m1_split_counts_unchanged(real_run: dict) -> None:
    rows = {name: totals["rows"] for name, totals in real_run["m1"]["totals"].items()}
    assert rows == M1_ROWS


def test_frozen_eval_report_matches_the_committed_one(real_run: dict) -> None:
    committed = json.loads(config.EVAL_FROZEN_REPORT_PATH.read_text(encoding="utf-8"))
    assert real_run["eval"] == committed


def test_frozen_eval_hash_identical_on_a_second_run(real_run: dict, tmp_path: Path) -> None:
    again = prepare_retrain(real_run["dir"], tmp_path, tmp_path / "eval_frozen.json")
    assert again["sha256"] == real_run["eval"]["sha256"]


def test_real_parts_share_no_sld(real_run: dict) -> None:
    assert set(real_run["split"]["shared_sld"].values()) == {0}
    rows = real_run["split"]["rows"]
    assert rows["train_fit"] + rows["validation"] == M1_ROWS["train"]
    assert rows["eval_frozen"] == real_run["eval"]["rows"]
