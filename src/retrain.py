"""One round of the retraining loop (KaizenML): phase in the next batch of held-out families.

1. Read the production pointer (models/production/latest.json): its round (0 for the M1 model)
   and phased_in_families.
2. Take the next batch from config.PHASE_IN_BATCHES; stop if every batch is phased in.
3. Train HistGradientBoostingClassifier with the fixed config.HGB_PARAMS on train_fit plus the
   phase-in pool rows of every phased-in family (earlier ones and this batch).
4. Run the promotion gate (src.gate) against the production model on the frozen eval set.
5. Log the run to MLflow (experiment config.MLFLOW_KAIZEN_EXPERIMENT_NAME).
6. Always write reports/retrain/round_<n>.json and round_<n>.md (the gate report). On promotion
   (unless --dry-run): save the new model in models/production/, delete the old model file, write
   latest.json and reports/production_model.json.

A rejected batch is retried unchanged by the next round: data and training are deterministic, so
only new features or parameters can change the outcome.

Needs data/processed/ (python -m src.prepare_data, then python -m src.prepare_retrain).
Run with: python -m src.retrain [--dry-run]
"""

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from src import config
from src.features import FEATURE_NAMES
from src.gate import decide, write_decision
from src.model_loader import file_sha256, load_model, resolve_model_path
from src.report import build_production_report
from src.train import new_version, portable_path, setup_mlflow

GATE_DECISION_FILE = "gate_decision.json"  # MLflow artifact name


# ---------------------------------------------------------------------------
# Round planning
# ---------------------------------------------------------------------------


def read_pointer(pointer: Path) -> dict[str, Any]:
    return json.loads(pointer.read_text(encoding="utf-8"))


def next_batch(phased_in: list[str]) -> list[str] | None:
    """The first batch of PHASE_IN_BATCHES not yet phased in, or None when all are."""
    for batch in config.PHASE_IN_BATCHES:
        done = [family for family in batch if family in phased_in]
        if len(done) == len(batch):
            continue
        if done:
            raise ValueError(f"batch {batch} is only partly phased in ({done}); fix latest.json")
        return batch
    return None


def load_parts(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """train_fit, phase_in_pool and eval_frozen from src.prepare_retrain."""
    parts = {}
    for name, path in (
        ("train_fit", config.TRAIN_FIT_PATH),
        ("phase_in_pool", config.PHASE_IN_POOL_PATH),
        ("eval_frozen", config.EVAL_FROZEN_PATH),
    ):
        file = processed_dir / path.name
        if not file.exists():
            raise FileNotFoundError(f"{file} is missing: run python -m src.prepare_retrain first")
        parts[name] = pd.read_parquet(file)
    return parts


def training_rows(train_fit: pd.DataFrame, pool: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """train_fit plus the phase-in pool rows of the given families."""
    return pd.concat([train_fit, pool[pool["family"].isin(families)]], ignore_index=True)


def fit_challenger(rows: pd.DataFrame, params: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**params)
    model.fit(rows[FEATURE_NAMES], (rows["label"] == "dga").astype(int))
    return model


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def flat_metrics(decision: dict) -> dict[str, float]:
    """The gate's numbers for both models, as MLflow metrics (None values left out)."""
    metrics: dict[str, float] = {"promoted": float(decision["promote"])}
    for side in ("champion", "challenger"):
        scores = decision["metrics"][side]
        for name in ("seen_f1", "new_family_recall", "prior_family_recall", "false_positive_rate"):
            if scores[name] is not None:
                metrics[f"{side}_{name}"] = scores[name]
        for name, value in scores["overall"].items():
            metrics[f"{side}_overall_{name}"] = value
        for part, values in scores.get("threshold_free", {}).items():
            for name, value in (values or {}).items():
                if value is not None:
                    metrics[f"{side}_{part}_{name}"] = value
    return metrics


THRESHOLD_FREE_ROWS = [
    ("overall", "all frozen rows"),
    ("new_families", "new families vs benign"),
    ("prior_families", "earlier families vs benign"),
]


def threshold_free_lines(decision: dict) -> list[str]:
    """The gate report's ROC-AUC / PR-AUC table (reported, not gated); empty for old decisions."""
    champion = decision["metrics"]["champion"].get("threshold_free")
    challenger = decision["metrics"]["challenger"].get("threshold_free")
    if not champion or not challenger:
        return []
    lines = [
        "",
        "Threshold-free metrics (reported, not gated):",
        "",
        "| Rows | Champion ROC-AUC | Challenger ROC-AUC | Champion PR-AUC | Challenger PR-AUC |",
        "|---|---|---|---|---|",
    ]
    for part, label in THRESHOLD_FREE_ROWS:
        if champion.get(part) is None or challenger.get(part) is None:
            continue
        c, h = champion[part], challenger[part]
        if None in (*c.values(), *h.values()):
            continue
        lines.append(
            f"| {label} | {c['roc_auc']:.4f} | {h['roc_auc']:.4f} | "
            f"{c['pr_auc']:.4f} | {h['pr_auc']:.4f} |"
        )
    return lines


def summary_markdown(report: dict) -> str:
    """The gate report as Markdown (the PR body, and the job summary in the workflow)."""
    decision = report["decision"]
    verdict = "**PROMOTE**" if decision["promote"] else "**REJECT**"
    lines = [
        f"## Retraining round {report['round']}: {verdict}",
        "",
        f"- New families: {', '.join(report['new_families'])}",
        f"- Earlier families: {', '.join(report['prior_families']) or 'none'}",
        f"- Training rows: {report['training_rows']} (train_fit {report['train_fit_rows']} + "
        f"phase-in pool {report['phase_in_rows']})",
        f"- Champion: `{report['champion']['version']}` ({report['champion']['model_type']}); "
        f"challenger: {config.RETRAIN_MODEL_TYPE} with fixed parameters",
        f"- Frozen eval set: {decision['eval_frozen']['rows']} rows, "
        f"sha256 `{decision['eval_frozen']['sha256'][:12]}`",
        f"- MLflow run: `{report['mlflow_run_id']}` "
        f"(experiment `{config.MLFLOW_KAIZEN_EXPERIMENT_NAME}`)",
        "",
        "| Criterion | Champion | Challenger | Change | Threshold | Result |",
        "|---|---|---|---|---|---|",
    ]
    for r in decision["reasons"]:
        delta = "" if r["delta"] is None else f"{r['delta']:+.4f}"
        champion = "" if r["champion"] is None else r["champion"]
        challenger = "" if r["challenger"] is None else r["challenger"]
        threshold = "" if r["threshold"] is None else r["threshold"]
        result = "pass" if r["passed"] else "**FAIL**"
        lines.append(
            f"| {r['name']} | {champion} | {challenger} | {delta} | {threshold} | {result} |"
        )
    lines += ["", "Reasons:", ""]
    lines += [f"- {'pass' if r['passed'] else 'FAIL'}: {r['message']}" for r in decision["reasons"]]
    families = sorted(set(report["new_families"]) | set(report["prior_families"]))
    champion = decision["metrics"]["champion"]["per_family_recall"]
    challenger = decision["metrics"]["challenger"]["per_family_recall"]
    lines += ["", "| Phased-in family | Champion recall | Challenger recall |", "|---|---|---|"]
    lines += [f"| {f} | {champion[f]:.4f} | {challenger[f]:.4f} |" for f in families]
    lines += threshold_free_lines(decision)
    if report.get("applied"):
        lines += [
            "",
            f"Promoted model: `{report['model']['version']}` "
            f"(sha256 `{report['model']['sha256'][:12]}`).",
        ]
    elif decision["promote"]:
        lines += ["", "Dry run: the gate would promote, but models/production/ was not changed."]
    return "\n".join(lines) + "\n"


def store_label() -> str:
    """How reports/production_model.json names the MLflow store holding the run."""
    return "github_actions_artifact" if os.environ.get("GITHUB_ACTIONS") == "true" else "local"


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def promote(
    model: HistGradientBoostingClassifier,
    production_dir: Path,
    old_pointer: dict,
    round_number: int,
    phased_in: list[str],
    decision: dict,
    run_id: str,
) -> dict:
    """Save the model next to latest.json, delete the old model file, write latest.json."""
    version = new_version(production_dir)
    model_path = production_dir / f"model_{version}.joblib"
    joblib.dump(model, model_path, compress=3)
    old_model = resolve_model_path(old_pointer["path"])
    if old_model.resolve().parent == production_dir.resolve() and old_model != model_path:
        old_model.unlink(missing_ok=True)  # rollback: revert the promotion PR (RUNBOOK.md)

    challenger = decision["metrics"]["challenger"]
    latest = {
        "path": portable_path(model_path),
        "version": version,
        "model_type": config.RETRAIN_MODEL_TYPE,
        "sha256": file_sha256(model_path),
        "mlflow_run_id": run_id,
        "sklearn_version": sklearn.__version__,
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "feature_names": list(FEATURE_NAMES),
        "round": round_number,
        "phased_in_families": phased_in,
        "eval_frozen_sha256": decision["eval_frozen"]["sha256"],
        "gate_metrics": {
            "seen_f1": challenger["seen_f1"],
            "false_positive_rate": challenger["false_positive_rate"],
            "overall_f1": challenger["overall"]["f1"],
        },
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (production_dir / "latest.json").write_text(json.dumps(latest, indent=2) + "\n", "utf-8")
    return latest


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------


def run_round(
    processed_dir: Path = config.PROCESSED_DIR,
    production_dir: Path = config.PRODUCTION_DIR,
    reports_dir: Path = config.RETRAIN_REPORTS_DIR,
    production_report_path: Path = config.PRODUCTION_REPORT_PATH,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
    dry_run: bool = False,
    hgb_params: dict | None = None,
    expected_sha256: str | None = None,
) -> dict | None:
    """Run one round. Returns the round report, or None when every batch is phased in."""
    pointer_path = production_dir / "latest.json"
    pointer = read_pointer(pointer_path)
    previous_round = int(pointer.get("round", 0))
    prior = list(pointer.get("phased_in_families", []))
    batch = next_batch(prior)
    if batch is None:
        print(f"Every batch is phased in ({prior}): nothing to retrain.")
        return None
    round_number = previous_round + 1
    print(f"Round {round_number}: phasing in {batch} (earlier: {prior or 'none'})")

    parts = load_parts(processed_dir)
    families = prior + batch
    rows = training_rows(parts["train_fit"], parts["phase_in_pool"], families)
    params = dict(config.HGB_PARAMS if hgb_params is None else hgb_params)
    champion, champion_meta = load_model(pointer_path)

    tracking_uri = tracking_uri or config.mlflow_tracking_uri()
    artifact_dir = artifact_dir or config.mlflow_artifact_dir()
    print(f"MLflow store: {tracking_uri}")
    setup_mlflow(tracking_uri, artifact_dir, config.MLFLOW_KAIZEN_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"round_{round_number}") as run:
        start = time.perf_counter()
        challenger = fit_challenger(rows, params)
        train_time = time.perf_counter() - start
        decision = decide(champion, challenger, parts["eval_frozen"], batch, prior, expected_sha256)

        mlflow.set_tags({"model_type": config.RETRAIN_MODEL_TYPE, "round": str(round_number)})
        mlflow.log_params(
            {
                **params,
                "model_type": config.RETRAIN_MODEL_TYPE,
                "round": round_number,
                "new_families": json.dumps(batch),
                "prior_families": json.dumps(prior),
                "champion_version": champion_meta["version"],
                "train_rows": len(rows),
                "train_fit_rows": len(parts["train_fit"]),
                "phase_in_rows": len(rows) - len(parts["train_fit"]),
                "eval_frozen_sha256": decision["eval_frozen"]["sha256"],
            }
        )
        mlflow.log_metrics({**flat_metrics(decision), "train_time_s": train_time})
        per_family = {"eval_frozen": decision["metrics"]["challenger"]["per_family_recall"]}
        mlflow.log_dict(per_family, config.PER_FAMILY_FILE)
        mlflow.log_dict(decision, GATE_DECISION_FILE)
        run_id = run.info.run_id

    applied = decision["promote"] and not dry_run
    report = {
        "round": round_number,
        "new_families": batch,
        "prior_families": prior,
        "promote": decision["promote"],
        "applied": applied,
        "dry_run": dry_run,
        "champion": {k: champion_meta.get(k) for k in ("version", "model_type", "sha256")},
        "model_type": config.RETRAIN_MODEL_TYPE,
        "params": params,
        "training_rows": len(rows),
        "train_fit_rows": len(parts["train_fit"]),
        "phase_in_rows": len(rows) - len(parts["train_fit"]),
        "mlflow_run_id": run_id,
        "mlflow_experiment": config.MLFLOW_KAIZEN_EXPERIMENT_NAME,
        "decision": decision,
    }
    if applied:
        latest = promote(
            challenger, production_dir, pointer, round_number, families, decision, run_id
        )
        report["model"] = {k: latest[k] for k in ("version", "sha256", "path")}
        stores = [(store_label(), tracking_uri)]
        production_report = build_production_report(pointer_path, stores)
        write_decision(production_report, production_report_path)

    reports_dir.mkdir(parents=True, exist_ok=True)
    write_decision(report, reports_dir / f"round_{round_number}.json")
    (reports_dir / f"round_{round_number}.md").write_text(summary_markdown(report), "utf-8")

    verdict = "PROMOTE" if decision["promote"] else "REJECT"
    print(f"Gate: {verdict}" + (" (dry run: models/production/ unchanged)" if dry_run else ""))
    for r in decision["reasons"]:
        print(f"  {'pass' if r['passed'] else 'FAIL'}  {r['name']}: {r['message']}")
    return report


def write_github_outputs(report: dict | None, reports_dir: Path) -> None:
    """Outputs for the retrain workflow (only when $GITHUB_OUTPUT is set)."""
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    values = {"promoted": "false", "round": "", "summary": ""}
    if report is not None:
        values = {
            "promoted": "true" if report["applied"] else "false",
            "round": str(report["round"]),
            "summary": (reports_dir / f"round_{report['round']}.md").as_posix(),
        }
    with open(output, "a", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="train, gate, log and write the round report, but never change models/production/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_round(dry_run=args.dry_run)
    write_github_outputs(report, config.RETRAIN_REPORTS_DIR)


if __name__ == "__main__":
    main()
