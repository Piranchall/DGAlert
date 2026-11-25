"""Static checks of .github/workflows/ci.yml: lint, test and build (the required checks), GHCR
publishing only on pushes to main and v* tags, and the Render deploy only on pushes to main."""

import re

import pytest
import yaml

from src import config

WORKFLOW_PATH = config.ROOT_DIR / ".github" / "workflows" / "ci.yml"
WORKFLOW = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
JOBS = WORKFLOW["jobs"]
DEPLOY_CONDITION = "github.event_name == 'push' && github.ref == 'refs/heads/main'"
LIVE_URL = "https://dgalert.onrender.com"
PUBLISH_CONDITION = (
    "github.event_name == 'push' && "
    "(github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v'))"
)


def triggers() -> dict:
    # PyYAML (YAML 1.1) reads the key `on` as the boolean True.
    return WORKFLOW.get("on", WORKFLOW.get(True))


def steps(job: str) -> list[dict]:
    return JOBS[job]["steps"]


def step(job: str, name: str) -> dict:
    return next(s for s in steps(job) if s.get("name") == name)


def uses(job: str) -> list[str]:
    return [s["uses"] for s in steps(job) if "uses" in s]


def test_jobs_named_as_the_required_checks() -> None:
    assert set(JOBS) == {"lint", "test", "build", "deploy"}
    # No display names: the check names stay exactly "lint", "test", "build" and "deploy".
    assert all("name" not in job for job in JOBS.values())


def test_build_waits_for_lint_and_test() -> None:
    assert sorted(JOBS["build"]["needs"]) == ["lint", "test"]
    assert "needs" not in JOBS["lint"]
    assert "needs" not in JOBS["test"]


def test_triggers() -> None:
    on = triggers()
    assert on["push"] == {"branches": ["main"], "tags": ["v*"]}
    assert "pull_request" in on


def test_runs_are_not_cancelled_on_main_or_tags() -> None:
    concurrency = WORKFLOW["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_lint_installs_only_the_pinned_ruff() -> None:
    install = step("lint", "Install ruff (version pinned in requirements.txt)")["run"]
    assert install == "pip install \"$(grep '^ruff==' requirements.txt)\""
    requirements = (config.ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^ruff==\S+$", requirements, re.MULTILINE)
    commands = [s.get("run", "") for s in steps("lint")]
    assert "ruff check src tests scripts" in commands
    assert "ruff format --check src tests scripts" in commands


def test_test_job_runs_pytest_on_python_313() -> None:
    assert WORKFLOW["env"]["PYTHON_VERSION"] == "3.13"
    assert "pytest -q" in [s.get("run", "") for s in steps("test")]
    assert "pip install -r requirements.txt" in step("test", "Install dependencies")["run"]


def test_only_build_can_write_packages() -> None:
    assert WORKFLOW["permissions"] == {"contents": "read"}
    assert JOBS["build"]["permissions"]["packages"] == "write"
    for job in ("lint", "test", "deploy"):
        assert "permissions" not in JOBS[job]


def test_image_name_is_lowercase() -> None:
    image = WORKFLOW["env"]["IMAGE"]
    assert image == "ghcr.io/piranchall/dgalert"
    assert image == image.lower()


def test_build_passes_git_sha_and_caches_layers() -> None:
    build = step("build", "Build image")["with"]
    assert build["build-args"] == "GIT_SHA=${{ steps.sha.outputs.short }}"
    assert build["load"] is True
    assert build["cache-from"] == "type=gha"
    assert "push" not in build  # pushing happens later, only for main and tags


def test_smoke_test_waits_with_a_timeout_and_calls_predict() -> None:
    wait = step("build", "Start container and wait until healthy")["run"]
    assert "seq 1 45" in wait
    assert "exit 1" in wait
    smoke = step("build", "Smoke test /health and /predict")["run"]
    assert "/predict" in smoke
    assert 'p["label"] == "dga"' in smoke
    assert 'h["git_sha"] == os.environ["EXPECTED_SHA"]' in smoke
    assert step("build", "Stop container")["if"] == "always()"


@pytest.mark.parametrize("name", ["Log in to GHCR", "Push the tested image to GHCR"])
def test_publishing_only_on_main_and_version_tags(name: str) -> None:
    assert step("build", name)["if"] == PUBLISH_CONDITION


def test_push_tags() -> None:
    push = step("build", "Push the tested image to GHCR")["run"]
    assert 'tags="$IMAGE:sha-$SHORT_SHA"' in push
    assert '$IMAGE:main"' in push
    assert "$IMAGE:$GITHUB_REF_NAME" in push
    assert 'docker tag dgalert:ci "$tag"' in push  # the image that passed the smoke test


def test_action_versions() -> None:
    all_uses = {u for job in JOBS for u in uses(job)}
    assert all_uses == {
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "docker/setup-buildx-action@v4",
        "docker/build-push-action@v7",
        "docker/login-action@v4",
    }


# ---------------------------------------------------------------------------
# deploy (Render)
# ---------------------------------------------------------------------------


def test_deploy_runs_after_build_only_on_pushes_to_main() -> None:
    deploy = JOBS["deploy"]
    assert deploy["needs"] == "build"
    assert deploy["if"] == DEPLOY_CONDITION
    assert "refs/tags" not in deploy["if"]  # version tags publish an image but do not deploy
    assert deploy["timeout-minutes"] <= 30


def test_deploy_hook_comes_from_the_secret_and_is_never_echoed() -> None:
    hook = step("deploy", "Trigger the Render deploy hook with the tested image")
    assert hook["env"] == {"RENDER_DEPLOY_HOOK": "${{ secrets.RENDER_DEPLOY_HOOK }}"}
    run = hook["run"]
    assert '[ -z "$RENDER_DEPLOY_HOOK" ]' in run  # a missing secret fails the job
    for line in run.splitlines():
        if "echo" in line:
            assert "$RENDER_DEPLOY_HOOK" not in line, line  # the name may appear, the value not
    assert "-o /dev/null" in run  # the response body is not printed either


def test_deploy_hook_deploys_the_tested_sha_tag() -> None:
    run = step("deploy", "Trigger the Render deploy hook with the tested image")["run"]
    assert 'image="$IMAGE:sha-${GITHUB_SHA::7}"' in run
    assert "urllib.parse.quote" in run
    assert r'case "$RENDER_DEPLOY_HOOK" in *\?*) sep="&" ;; *) sep="?" ;; esac' in run
    assert '"${RENDER_DEPLOY_HOOK}${sep}imgURL=${encoded}"' in run


def test_deploy_waits_for_the_live_git_sha_with_a_timeout() -> None:
    run = step("deploy", "Wait until the live service reports this commit")["run"]
    assert 'expected="${GITHUB_SHA::7}"' in run
    assert '"$RENDER_URL/health"' in run
    assert "seq 1 60" in run
    assert "exit 1" in run


def test_live_url_in_workflow_and_readme() -> None:
    assert WORKFLOW["env"]["RENDER_URL"] == LIVE_URL
    readme = (config.ROOT_DIR / "README.md").read_text(encoding="utf-8")
    assert "## Deployment" in readme
    assert "curl.exe" in readme
    for endpoint in ("/health", "/predict", "/explain", "/stats"):
        assert f"{LIVE_URL}{endpoint}" in readme
