"""Static checks of .github/workflows/retrain.yml: manual real rounds, weekly dry runs, the frozen
eval report kept as the reference, and the promotion PR opened with RETRAIN_PR_TOKEN only."""

import re

import yaml

from src import config

WORKFLOW = yaml.safe_load(
    (config.ROOT_DIR / ".github" / "workflows" / "retrain.yml").read_text(encoding="utf-8")
)
JOB = WORKFLOW["jobs"]["retrain"]
PR_STEP = "Open the promotion PR"


def triggers() -> dict:
    # PyYAML (YAML 1.1) reads the key `on` as the boolean True.
    return WORKFLOW.get("on", WORKFLOW.get(True))


def step(name: str) -> dict:
    return next(s for s in JOB["steps"] if s.get("name") == name)


def names() -> list[str]:
    return [s.get("name") or s["uses"] for s in JOB["steps"]]


def test_manual_dispatch_defaults_to_a_dry_run_and_the_schedule_is_weekly() -> None:
    on = triggers()
    dry_run = on["workflow_dispatch"]["inputs"]["dry_run"]
    assert dry_run["type"] == "boolean" and dry_run["default"] is True
    (schedule,) = on["schedule"]
    assert re.fullmatch(r"\d+ \d+ \* \* \d", schedule["cron"])  # once a week


def test_schedule_is_always_a_dry_run() -> None:
    # Only a manual dispatch with dry_run unticked is a real round.
    assert WORKFLOW["env"]["DRY_RUN"] == (
        "${{ github.event_name != 'workflow_dispatch' || inputs.dry_run }}"
    )
    assert '"$DRY_RUN" = "true"' in step("Retrain and run the gate")["run"]
    assert "python -m src.retrain --dry-run" in step("Retrain and run the gate")["run"]


def test_real_rounds_only_on_main() -> None:
    guard = step("Real rounds only on main")
    assert guard["if"] == "env.DRY_RUN == 'false' && github.ref != 'refs/heads/main'"
    assert "exit 1" in guard["run"]
    assert names().index("Real rounds only on main") == 0


def test_permissions_and_concurrency() -> None:
    assert WORKFLOW["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "permissions" not in JOB  # writes happen only through RETRAIN_PR_TOKEN
    assert WORKFLOW["concurrency"] == {"group": "retrain", "cancel-in-progress": False}
    assert JOB["timeout-minutes"] <= 60


def test_steps_in_order() -> None:
    order = names()
    for earlier, later in [
        ("Install dependencies", "Skip if this round's PR is already open"),
        ("Skip if this round's PR is already open", "Prepare data"),
        ("Prepare data", "Retrain and run the gate"),
        ("Retrain and run the gate", "Upload the MLflow store and the round report"),
        ("Upload the MLflow store and the round report", PR_STEP),
    ]:
        assert order.index(earlier) < order.index(later), (earlier, later)


def test_frozen_eval_report_is_not_overwritten() -> None:
    run = step("Prepare data")["run"]
    assert "python -m src.prepare_data" in run
    assert 'python -m src.prepare_retrain --eval-report "$RUNNER_TEMP/eval_frozen.json"' in run


def test_mlflow_store_and_report_always_uploaded() -> None:
    upload = step("Upload the MLflow store and the round report")
    assert upload["uses"] == "actions/upload-artifact@v7"
    assert upload["if"].startswith("always()")
    assert upload["with"]["path"].split() == ["mlflow_local/", "reports/retrain/"]


def test_open_round_pr_is_not_duplicated() -> None:
    run = step("Skip if this round's PR is already open")["run"]
    assert 'gh pr list --head "retrain/round-$round" --state open' in run
    for name in ("Prepare data", "Retrain and run the gate"):
        assert step(name)["if"] == "steps.pending.outputs.skip != 'true'"


def test_pr_only_on_a_real_promotion_with_the_token() -> None:
    pr = step(PR_STEP)
    assert pr["if"] == "steps.retrain.outputs.promoted == 'true' && env.DRY_RUN == 'false'"
    assert pr["env"]["GH_TOKEN"] == "${{ secrets.RETRAIN_PR_TOKEN }}"
    for s in JOB["steps"]:
        if s.get("name") != PR_STEP:
            assert "RETRAIN_PR_TOKEN" not in str(s.get("env", {})), s.get("name")
    checkout = next(s for s in JOB["steps"] if s.get("uses", "").startswith("actions/checkout"))
    assert checkout["with"]["persist-credentials"] is False


def test_pr_contents_title_and_reviewer() -> None:
    run = step(PR_STEP)["run"]
    assert '[ -z "$GH_TOKEN" ]' in run
    assert 'branch="retrain/round-$ROUND"' in run
    assert '--title "retrain: round $ROUND (automated)"' in run
    assert '--body-file "$SUMMARY"' in run
    assert '--reviewer "$reviewer"' in run
    assert 'user.name="github-actions[bot]"' in run
    for path in (
        "models/production",
        '"reports/retrain/round_$ROUND.json"',
        '"reports/retrain/round_$ROUND.md"',
        "reports/production_model.json",
    ):
        assert path in run
    # The reviewer is the member who does not own the token.
    assert WORKFLOW["env"]["MEMBERS"].split() == ["Piranchall", "hmazhar008"]
    assert 'if [ "$member" != "$owner" ]; then reviewer="$member"; fi' in run


def test_no_expressions_inside_shell_scripts() -> None:
    # Inputs and outputs reach the shell only through env, never as ${{ }} inside run:.
    for s in JOB["steps"]:
        assert "${{" not in s.get("run", ""), s.get("name")


def test_action_versions_match_ci() -> None:
    uses = {s["uses"] for s in JOB["steps"] if "uses" in s}
    assert uses == {"actions/checkout@v7", "actions/setup-python@v7", "actions/upload-artifact@v7"}
