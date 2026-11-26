"""The README's "Video demo walkthrough" stays in step with the rest of the README and the repo.

Commands that also appear in Quick start and Deployment must be the same text, every step with
commands gives them for PowerShell and bash, and the links and workflow inputs it names exist.
"""

import re

import pytest
import yaml

from src import config

README = (config.ROOT_DIR / "README.md").read_text(encoding="utf-8")
SECTION_TITLE = "Video demo walkthrough"
STEPS = [
    "1. Clean start",
    "2. Docker health",
    "3. Monitoring",
    "4. CI",
    "5. Deployment",
    "6. Rollback",
    "7. Retraining",
    "8. Cleanup",
]
NO_COMMANDS = {"6. Rollback"}  # a link to RUNBOOK.md only


def section(title: str, level: str = "##", text: str = README) -> str:
    """The text of a section, up to the next heading of the same level (or the end of text)."""
    pattern = rf"^{level} {re.escape(title)}\s*$(.*?)(?=^{level} |\Z)"
    match = re.search(pattern, text, re.M | re.S)
    assert match, f"section {level} {title!r} not found"
    return match.group(1)


def blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"^```{language}\n(.*?)^```", text, re.M | re.S)


def lines(text: str, language: str | None = None) -> list[str]:
    languages = [language] if language else ["bash", "powershell"]
    return [
        line.strip()
        for lang in languages
        for block in blocks(text, lang)
        for line in block.splitlines()
        if line.strip()
    ]


WALKTHROUGH = section(SECTION_TITLE)
QUICK_START = section("Quick start")
DEPLOYMENT = section("Deployment")


def step(name: str) -> str:
    """One step, searched inside the walkthrough only (the last step ends with the section)."""
    return section(name, "###", WALKTHROUGH)


def test_section_comes_right_after_quick_start_and_is_in_contents() -> None:
    top_level = re.findall(r"^## (.+?)\s*$", README, re.M)
    assert top_level[top_level.index("Quick start") + 1] == SECTION_TITLE
    contents = section("Contents")
    assert "[Quick start](#quick-start)\n4. [Video demo walkthrough](#video-demo-walkthrough)" in (
        contents
    )


def test_steps_are_in_the_video_order() -> None:
    assert re.findall(r"^### (.+?)\s*$", WALKTHROUGH, re.M) == STEPS


@pytest.mark.parametrize("name", [s for s in STEPS if s not in NO_COMMANDS])
def test_every_step_has_a_powershell_and_a_bash_block(name: str) -> None:
    text = step(name)
    assert len(blocks(text, "powershell")) == 1, name
    assert len(blocks(text, "bash")) == 1, name


def test_clean_start_reuses_the_quick_start_commands() -> None:
    quick = set(lines(QUICK_START))
    reused = [
        line
        for line in lines(step("1. Clean start"))
        if line.startswith(("git clone", "cd ", "docker build", "docker run", "curl"))
    ]
    # clone, cd, build, run and the two local calls, once per shell
    assert len(reused) == 12
    for line in reused:
        assert line in quick, f"not the same as in Quick start: {line}"


def test_deployment_step_reuses_the_deployment_section_calls() -> None:
    deployment = set(lines(DEPLOYMENT))
    live = [line for line in lines(step("5. Deployment")) if "onrender.com" in line]
    assert len(live) == 8  # health, predict, explain, stats, in each shell
    for line in live:
        assert line in deployment, f"not the same as in Deployment: {line}"
    for language in ("powershell", "bash"):
        assert "git rev-parse --short=7 origin/main" in lines(step("5. Deployment"), language)


def test_local_calls_use_127_0_0_1_and_powershell_uses_curl_exe() -> None:
    assert "localhost" not in WALKTHROUGH
    for line in lines(WALKTHROUGH, "powershell"):
        assert not re.search(r"(^|[|{;]\s*)curl\s", line), line
    for line in lines(WALKTHROUGH, "bash"):
        assert "curl.exe" not in line, line


def test_container_name_and_image_match_the_docker_run_line() -> None:
    run = next(line for line in lines(QUICK_START) if line.startswith("docker run"))
    name = re.search(r"--name (\S+)", run).group(1)
    for line in lines(WALKTHROUGH):
        if line.startswith(("docker inspect", "docker ps", "docker rm")) or "inspect -f" in line:
            assert re.search(rf"\b{name}\b", line), line


def test_monitoring_counts_match_the_calls() -> None:
    text = step("3. Monitoring")
    for language in ("powershell", "bash"):
        commands = lines(text, language)
        assert sum("moiqidfvjjhjmndqp.in" in c for c in commands) == 1  # in a loop of three
        assert any(re.search(r"1\.\.3|for i in 1 2 3", c) for c in commands)
        assert sum("google.com" in c for c in commands) == 1
    # with step 1's /predict call: 5 calls, 4 dga and 1 benign
    assert "`count` 5" in text
    assert "`dga` 4, `benign` 1" in text
    assert "[`GET /stats`](#get-stats)" in text
    assert re.search(r"^### `GET /stats`", README, re.M)


def test_ci_demo_is_a_closed_draft_and_needs_push_access() -> None:
    text = step("4. CI")
    assert "needs push access" in text
    assert "own\nfork with Actions enabled" in text or "own fork with Actions enabled" in text
    assert "**never merged**" in text
    powershell, bash = lines(text, "powershell"), lines(text, "bash")
    for commands in (powershell, bash):
        joined = "\n".join(commands)
        assert "gh pr create --draft --base main" in joined
        assert "(do not merge)" in joined
        assert "--delete-branch" in joined
        assert "gh pr merge" not in joined
    writer_ps = next(c for c in powershell if c.startswith("py -3 -c "))
    writer_bash = next(c for c in bash if c.startswith("python3 -c "))
    assert writer_ps.removeprefix("py -3 -c ") == writer_bash.removeprefix("python3 -c ")


def test_ci_demo_matches_the_ci_jobs() -> None:
    workflow = yaml.safe_load(
        (config.ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    assert {"lint", "test", "build"} <= set(jobs)
    assert set(jobs["build"]["needs"]) >= {"lint", "test"}  # so a failing test skips build
    assert "lint passes, test fails, build is skipped" in step("4. CI")


def test_rollback_links_the_runbook_and_its_command() -> None:
    text = step("6. Rollback")
    assert "[RUNBOOK.md](RUNBOOK.md)" in text
    runbook = (config.ROOT_DIR / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "gh workflow run deploy.yml -f tag=sha-" in runbook
    assert "gh workflow run deploy.yml -f tag=sha-" in text


def test_retraining_names_the_round_1_run_and_a_real_dry_run_input() -> None:
    text = step("7. Retraining")
    assert "https://github.com/Piranchall/DGAlert/actions/runs/45668715690" in text
    # the same run the README's Project Focus cites for the committed round 1 report
    assert "Actions run 45668715690" in section("Project Focus")
    for language in ("powershell", "bash"):
        assert lines(text, language) == ["gh workflow run retrain.yml -f dry_run=true"]
    workflow = yaml.safe_load(
        (config.ROOT_DIR / ".github" / "workflows" / "retrain.yml").read_text(encoding="utf-8")
    )
    triggers = workflow.get("on", workflow.get(True))
    assert "dry_run" in triggers["workflow_dispatch"]["inputs"]


def test_cleanup_removes_the_container() -> None:
    for language in ("powershell", "bash"):
        assert lines(step("8. Cleanup"), language) == ["docker rm -f dgalert"]
