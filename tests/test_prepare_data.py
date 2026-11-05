"""Tests for src.prepare_data on a tiny synthetic sample written to tmp_path (never data/raw)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src.features import FEATURE_NAMES
from src.prepare_data import prepare_data

SHARED = "shared-site.com"  # on both sides: must be removed from both
TOO_LONG_LABEL = "a" * 64 + ".com"  # label longer than 63
TOO_LONG_TOTAL = ("b" * 60 + ".") * 4 + "example.com"  # every label ok, total > 253

TRAIN_A = ["qwxkzt.com", "plmnbv.net", "zxcvbq.org", "hjkwer.com", "tyuqop.net", "rtyfgh.org"]
TRAIN_B = ["mnbvxz.com", "lkjpoi.net", "ghjtyr.org", "vbnmqw.com", "asdqwe.net", "fghrty.org"]
HELD = ["heldone.com", "heldtwo.com", "heldthree.com"]
# Held-out domains reusing every training sld with another TLD: the ones whose sld lands in train
# must be dropped from test_future.
HELD_SHARED_SLD = [d.split(".")[0] + ".info" for d in TRAIN_A + TRAIN_B]
EXCLUDED = ["excludedone.com", "excludedtwo.com"]
BENIGN = [
    "google.com",
    "mail.google.com",  # same sld as google.com: must stay in the same split
    "wikipedia.org",
    "stackoverflow.com",
    "github.com",
    "python.org",
    "bbc.co.uk",
    "example.com",
    "mozilla.org",
    "apache.org",
    "debian.org",
    "kernel.org",
    "nature.com",
    "weather.gov",
    "openstreetmap.org",
    "archive.org",
]
BAD_DGA = [
    ("", "fam_a"),  # empty domain
    (None, "fam_a"),  # null domain
    ("valid-but-no-family.com", None),  # null family
    ("bad_domain!.com", "fam_a"),  # invalid characters
    (TOO_LONG_LABEL, "fam_a"),
    (TOO_LONG_TOTAL, "fam_b"),
    ("QWXKZT.com", "fam_a"),  # duplicate of qwxkzt.com after normalization
]
BAD_BENIGN = ["co.uk", "_wildcard_.ph", "https://Google.com/"]  # bare suffix, invalid, duplicate


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Write a tiny umudga + tranco sample in the same format as data/sample/."""
    rows = (
        [(d, "fam_a") for d in TRAIN_A]
        + [(d, "fam_b") for d in TRAIN_B]
        + [(d, "fam_held") for d in HELD + HELD_SHARED_SLD]
        + [(d, "fam_excl") for d in EXCLUDED]
        + [(SHARED, "fam_b")]
        + BAD_DGA
    )
    out = tmp_path / "sample"
    out.mkdir()
    pd.DataFrame(rows, columns=["domain", "family"]).to_csv(
        out / config.UMUDGA_SAMPLE_PATH.name, index=False
    )
    benign = BENIGN + [SHARED] + BAD_BENIGN
    pd.DataFrame({"domain": benign, "tranco_rank": range(1, len(benign) + 1)}).to_csv(
        out / config.TRANCO_SAMPLE_PATH.name, index=False
    )
    return out


@pytest.fixture
def families(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "HELD_OUT_FAMILIES", ["fam_held"])
    monkeypatch.setattr(config, "EXCLUDED_FAMILIES", ["fam_excl"])


@pytest.fixture
def prepared(sample_dir: Path, tmp_path: Path, families: None) -> tuple[dict, dict]:
    """Run prepare_data once; return (report, {split name: frame})."""
    out = tmp_path / "processed"
    report = prepare_data(sample_dir, out)
    splits = {
        name: pd.read_parquet(out / f"{name}.parquet")
        for name in ("train", "test_seen", "test_future")
    }
    return report, splits


def all_rows(splits: dict) -> pd.DataFrame:
    return pd.concat(splits.values(), ignore_index=True)


def test_writes_all_outputs(prepared: tuple, tmp_path: Path) -> None:
    report, _ = prepared
    out = tmp_path / "processed"
    for name in ("train.parquet", "test_seen.parquet", "test_future.parquet"):
        assert (out / name).is_file()
    assert json.loads((out / "split_report.json").read_text(encoding="utf-8")) == report


def test_no_nulls_and_no_duplicate_domains(prepared: tuple) -> None:
    _, splits = prepared
    for name, frame in splits.items():
        assert not frame.isna().any().any(), f"nulls in {name}"
        assert not frame["domain"].duplicated().any(), f"duplicate domains in {name}"


def test_invalid_and_empty_label_rows_removed(prepared: tuple) -> None:
    report, splits = prepared
    domains = set(all_rows(splits)["domain"])
    for bad in ["", "bad_domain!.com", "_wildcard_.ph", "co.uk", TOO_LONG_LABEL, TOO_LONG_TOTAL]:
        assert bad not in domains
    drops = report["validation_drops"]
    assert drops["null_or_empty"] == 3  # empty domain, null domain, null family
    assert drops["invalid_characters"] == 2  # bad_domain!.com, _wildcard_.ph
    assert drops["empty_second_level_label"] == 1  # co.uk
    assert drops["length_out_of_range"] == 2
    assert drops["duplicate_domain"] == 2  # QWXKZT.com and https://Google.com/


def test_domain_on_both_sides_removed_from_both(prepared: tuple) -> None:
    report, splits = prepared
    assert SHARED not in set(all_rows(splits)["domain"])
    assert report["validation_drops"]["on_both_sides"] == 2  # one benign row + one dga row


def test_excluded_family_appears_nowhere(prepared: tuple) -> None:
    report, splits = prepared
    rows = all_rows(splits)
    assert "fam_excl" not in set(rows["family"])
    assert not set(EXCLUDED) & set(rows["domain"])
    assert report["validation_drops"]["excluded_family"] == len(EXCLUDED)


def test_no_sld_shared_between_train_and_test_seen(prepared: tuple) -> None:
    report, splits = prepared
    assert not set(splits["train"]["sld"]) & set(splits["test_seen"]["sld"])
    assert report["shared_sld_train_test_seen"] == 0
    # google.com and mail.google.com share the sld "google": same split
    train_domains = set(splits["train"]["domain"])
    assert ("google.com" in train_domains) == ("mail.google.com" in train_domains)


def test_held_out_family_only_in_test_future(prepared: tuple) -> None:
    _, splits = prepared
    assert "fam_held" not in set(splits["train"]["family"])
    assert "fam_held" not in set(splits["test_seen"]["family"])
    assert set(splits["test_future"]["domain"]) >= set(HELD)


def test_test_future_has_no_sld_seen_in_train(prepared: tuple) -> None:
    report, splits = prepared
    train_slds = set(splits["train"]["sld"])
    assert not train_slds & set(splits["test_future"]["sld"])
    assert report["shared_sld_train_test_future"] == 0
    # Exactly the held-out rows whose sld is in train were dropped, and the fixture has some.
    expected = sum(d.split(".")[0] in train_slds for d in HELD_SHARED_SLD)
    assert expected > 0
    assert report["held_out_dropped_sld_in_train"] == expected
    # Held-out rows whose sld is not in train are kept.
    kept = {d for d in HELD_SHARED_SLD if d.split(".")[0] not in train_slds}
    assert kept <= set(splits["test_future"]["domain"])


def test_test_future_is_held_out_dga_plus_test_seen_benign(prepared: tuple) -> None:
    _, splits = prepared
    future, seen = splits["test_future"], splits["test_seen"]
    dga = future[future["label"] == "dga"]
    assert set(dga["family"]) == {"fam_held"}
    benign = future[future["label"] == "benign"]
    seen_benign = seen[seen["label"] == "benign"]
    assert len(seen_benign) > 0, "fixture should put some benign rows in test_seen"
    assert set(benign["domain"]) == set(seen_benign["domain"])
    assert set(future["family"]) <= {"fam_held", "benign"}


def test_columns_and_feature_order(prepared: tuple) -> None:
    _, splits = prepared
    for frame in splits.values():
        assert list(frame.columns) == ["domain", "sld", "family", "label", *FEATURE_NAMES]
        assert set(frame["label"]) <= {"benign", "dga"}


def test_labels_match_families(prepared: tuple) -> None:
    rows = all_rows(prepared[1])
    assert (rows.loc[rows["family"] == "benign", "label"] == "benign").all()
    assert (rows.loc[rows["family"] != "benign", "label"] == "dga").all()


def test_running_twice_gives_identical_outputs(
    sample_dir: Path, tmp_path: Path, families: None
) -> None:
    first = prepare_data(sample_dir, tmp_path / "run1")
    second = prepare_data(sample_dir, tmp_path / "run2")
    assert first == second
    for name in ("train", "test_seen", "test_future"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "run1" / f"{name}.parquet"),
            pd.read_parquet(tmp_path / "run2" / f"{name}.parquet"),
        )
