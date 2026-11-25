"""Static checks of .github/workflows/deploy.yml (deploy one sha- image tag to Render, called by CI
and run by hand for rollbacks) and of the RUNBOOK commands that run it."""

import re

import pytest
import yaml

from src import config

WORKFLOW_DIR = config.ROOT_DIR / ".github" / "workflows"
WORKFLOW_FILE = "deploy.yml"
WORKFLOW = yaml.safe_load((WORKFLOW_DIR / WORKFLOW_FILE).read_text(encoding="utf-8"))
JOB = WORKFLOW["jobs"]["deploy"]
LIVE_URL = "https://dgalert.onrender.com"
RUNBOOK = (config.ROOT_DIR / "RUNBOOK.md").read_text(encoding="utf-8")


def triggers() -> dict:
    # PyYAML (YAML 1.1) reads the key `on` as the boolean True.
    return WORKFLOW.get("on", WORKFLOW.get(True))


def step(name: str) -> dict:
    return next(s for s in JOB["steps"] if s.get("name") == name)


def tag_pattern() -> re.Pattern:
    """The regex the format check applies to the tag, as written in the workflow (bash =~)."""
    run = step("Check the tag format")["run"]
    (pattern,) = re.findall(r"=~ \^(\S+)\$ \]\]", run)
    return re.compile(rf"\A{pattern}\Z")


def test_triggers_manual_and_from_ci_with_a_required_tag() -> None:
    on = triggers()
    assert set(on) == {"workflow_dispatch", "workflow_call"}
    for trigger in ("workflow_dispatch", "workflow_call"):
        tag = on[trigger]["inputs"]["tag"]
        assert tag["required"] is True
        assert tag["type"] == "string"
    assert on["workflow_call"]["secrets"] == {"RENDER_DEPLOY_HOOK": {"required": True}}


def test_one_job_no_write_permissions_and_a_timeout() -> None:
    assert set(WORKFLOW["jobs"]) == {"deploy"}
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert "permissions" not in JOB
    assert JOB["timeout-minutes"] <= 30


def test_deploys_run_one_at_a_time() -> None:
    # Repo-wide group, shared by manual runs and CI's calls: queue, never cancel a deploy.
    assert JOB["concurrency"] == {"group": "render-deploy", "cancel-in-progress": False}


def test_image_and_live_url() -> None:
    assert WORKFLOW["env"]["IMAGE"] == "ghcr.io/piranchall/dgalert"
    assert WORKFLOW["env"]["RENDER_URL"] == LIVE_URL


def test_tag_reaches_the_scripts_only_through_env() -> None:
    assert JOB["env"]["TAG"] == "${{ inputs.tag }}"
    for s in JOB["steps"]:
        assert "${{" not in s.get("run", ""), s["name"]  # no expression injection into shell


@pytest.mark.parametrize("tag", ["sha-4e822c5", "sha-50947c7", "sha-0000000"])
def test_tag_format_accepts(tag: str) -> None:
    assert tag_pattern().match(tag)


@pytest.mark.parametrize(
    "tag",
    [
        "4e822c5",  # no prefix
        "sha-4E822C5",  # uppercase
        "sha-4e822c",  # too short
        "sha-4e822c5a",  # too long
        "main",  # moving tag, would not identify a commit
        "v2.0-milestone2",
        "sha-4e822c5;rm",
        "sha-4e822c5\nfoo",
        "",
    ],
)
def test_tag_format_rejects(tag: str) -> None:
    assert not tag_pattern().match(tag)


def test_checks_the_image_exists_before_calling_render() -> None:
    names = [s["name"] for s in JOB["steps"]]
    assert names.index("Check that the image exists in GHCR") < names.index(
        "Trigger the Render deploy hook with this image"
    )
    run = step("Check that the image exists in GHCR")["run"]
    assert 'docker manifest inspect "$IMAGE:$TAG"' in run
    assert "exit 1" in run


def test_hook_comes_from_the_secret_and_is_never_echoed() -> None:
    hook = step("Trigger the Render deploy hook with this image")
    assert hook["env"] == {"RENDER_DEPLOY_HOOK": "${{ secrets.RENDER_DEPLOY_HOOK }}"}
    run = hook["run"]
    assert '[ -z "$RENDER_DEPLOY_HOOK" ]' in run  # a missing secret fails the job
    for line in run.splitlines():
        if "echo" in line:
            assert "$RENDER_DEPLOY_HOOK" not in line, line  # the name may appear, the value not
    assert "-o /dev/null" in run  # the response body is not printed either


def test_hook_deploys_the_given_tag() -> None:
    run = step("Trigger the Render deploy hook with this image")["run"]
    assert 'image="$IMAGE:$TAG"' in run
    assert "urllib.parse.quote" in run
    assert r'case "$RENDER_DEPLOY_HOOK" in *\?*) sep="&" ;; *) sep="?" ;; esac' in run
    assert '"${RENDER_DEPLOY_HOOK}${sep}imgURL=${encoded}"' in run


def test_waits_for_the_tag_commit_on_the_live_service() -> None:
    run = step("Wait until the live service reports this commit")["run"]
    assert 'expected="${TAG#sha-}"' in run
    assert '"$RENDER_URL/health"' in run
    assert "seq 1 60" in run
    assert "exit 1" in run


# ---------------------------------------------------------------------------
# README and RUNBOOK
# ---------------------------------------------------------------------------


def test_readme_deployment_section() -> None:
    readme = (config.ROOT_DIR / "README.md").read_text(encoding="utf-8")
    assert "## Deployment" in readme
    assert "RUNBOOK.md" in readme
    assert "curl.exe" in readme
    for endpoint in ("/health", "/predict", "/explain", "/stats"):
        assert f"{LIVE_URL}{endpoint}" in readme


def test_runbook_commands_match_the_workflow() -> None:
    commands = re.findall(r"gh workflow run (\S+) -f (\w+)=(\S+)", RUNBOOK)
    assert commands, "RUNBOOK.md has no gh workflow run command"
    inputs = triggers()["workflow_dispatch"]["inputs"]
    for file, input_name, value in commands:
        assert file == WORKFLOW_FILE
        assert input_name in inputs
        assert value.startswith("sha-") or value == "$tag", value
    # $tag is always built as "sha-" + the short commit.
    assert '$tag = "sha-" + (git rev-parse --short=7 HEAD)' in RUNBOOK


def test_runbook_covers_the_required_topics() -> None:
    for topic in (
        "## Normal redeploy",
        "## Roll back the service",
        "## Roll forward",
        "## Roll back a model",
        "## Run a production image locally",
        "## Free tier: cold starts and `/stats`",
        "## When the health check fails",
        "## Rehearsal record",
    ):
        assert topic in RUNBOOK, topic
    assert LIVE_URL in RUNBOOK
    # Reverting S1's PR would delete the only production model: the RUNBOOK must say not to.
    assert "Do not revert #30" in RUNBOOK
