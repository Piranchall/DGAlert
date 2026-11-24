# DGAlert: instructions for Claude Code

DGAlert classifies a domain name as `benign` or `dga` from the string alone. MLOps course project
(two members: Piranchal 29050, Muhammad Hussain 28985). Milestone 1 is tagged `v1.0-milestone1`.
We are now doing Milestone 2 (Docker, CI/CD, deployment, monitoring) plus the KaizenML retraining loop.

The work is split into **slices**, defined in `docs/M2_SLICES.md`. When asked to "do slice X",
read that slice in full and follow the workflow below.

## Environment

- Python **3.13**, virtualenv in `venv/` at the repo root (gitignored).
- Windows (PowerShell): `venv\Scripts\python -m <module>`. macOS/Linux: `venv/bin/python -m <module>`.
- Always call tools through the venv interpreter (`python -m pytest`, `python -m ruff`), never a global one.

## Commands

| Task | Command (use the venv python) |
|---|---|
| Install | `python -m pip install -r requirements.txt` |
| Prepare data | `python -m src.prepare_data` (writes `data/processed/`) |
| Train (M1) | `python -m src.train --model all` (about 11 min; avoid unless the slice needs it) |
| Lint | `python -m ruff check src tests scripts` and `python -m ruff format --check src tests scripts` |
| Tests | `python -m pytest -q` (sanity tests: `python -m pytest -m sanity`) |
| Serve | `python -m uvicorn src.app:app` (docs at http://127.0.0.1:8000/docs) |

## Identity (both members share one laptop)

Every slice has an owner in `docs/M2_SLICES.md`. Commits, pushes and the PR must come from that owner:

| Owner | GitHub account (`gh api user --jq .login`) | `git config user.name` | `git config user.email` |
|---|---|---|---|
| Piranchal | `Piranchall` | `Piranchall` | `piranchalmukesh@gmail.com` |
| Hussain | `hmazhar008` | `Muhammad Hussain` | `m.hussain.28985@iba.edu.pk` |

Check all three values at the start of the slice; the command block you hand over checks them again before committing. If any value does not match
the slice owner, stop and tell the user which identity is active and which one the slice needs.
Never run `gh auth switch` or `git config user.*` yourself; the user switches identity.

## Workflow for every slice (git first, git last)

1. **Git first.** Run the identity check above. Then `git switch main`, `git pull --ff-only`, and
   `git switch -c <branch from the slice>`.
   Stop if the working tree is dirty and ask what to do.
2. **Plan before editing.** Read the slice, the files it touches and their tests. Present a short plan
   (files to change, design choices, tests to add, how each acceptance criterion will be checked)
   and wait for approval.
3. **Stay inside the slice.** Do not refactor or "fix" things outside its scope. List anything else you
   notice under "Out of scope findings" in the PR description instead.
4. **Tests with every behaviour change.** New behaviour gets a pytest test. Tests must not need the
   raw data or network access.
5. **The user commits, pushes and opens the PR, not you.** Never run `git add`, `git commit`,
   `git push`, `gh pr create` or `gh pr merge`. Creating the branch is fine.
6. **Git last: hand over the commands.** When the work is done:
   - Run ruff check, ruff format --check and the full pytest suite; all must pass. Report the results.
   - Write the PR description to `$env:TEMP\pr_<slice ID>.md` (outside the repo, so it is never
     committed), following `.github/pull_request_template.md`: slice ID, summary, acceptance
     criteria as a checklist with the evidence for each (command output, numbers), out-of-scope
     findings, and `Closes #<issue>` for the slice's issue.
   - Then print one PowerShell block for the user to run, in this order:
     1. the identity check (`gh api user --jq .login; git config user.name; git config user.email`);
     2. small commits with conventional messages (`feat:`, `fix:`, `infra:`, `docs:`, `test:`,
        `chore:`), each `git add` listing explicit file paths (never `git add .` or `-A`);
     3. `git log main..HEAD --format="%an <%ae> %s"` to confirm every commit has the owner's name
        and email;
     4. `git push -u origin <branch>`;
     5. `gh pr create --base main --title "<slice ID>: <title>" --body-file $env:TEMP\pr_<slice ID>.md --reviewer <the other member's GitHub account>`.
   - Never merge, never push to main, never force-push.

## Hard rules

- Do not change M1 results: `prepare_data` must keep producing train 153,627 / test_seen 38,373 /
  test_future 61,141 rows, and `src/features.py` feature values must not change, unless the slice
  explicitly says so.
- Never commit `data/raw/`, `data/processed/`, `venv/`, secrets or `.env`.
- Until slice S2 is merged, running training or `src.report` modifies `mlflow.db`, `mlruns/` and
  `reports/`. Do not commit those changes unless the slice says so; restore them with
  `git restore mlflow.db reports` and remove new run folders under `mlruns/` (check with
  `git clean -n mlruns` first).
- Never commit files under `models/` except `models/production/` (after S1).
- Docker images and GHCR names must be lowercase: `ghcr.io/piranchall/dgalert`.
- Keep `README.md` accurate: if a slice changes a command, endpoint or number, update the README in
  the same PR.
- Do not write the per-member reflections in `CONTRIBUTION.md`; each member writes their own.