# DGAlert: Milestone 2 slice plan

Each slice is one GitHub issue, one branch, one pull request, reviewed and merged by the **other**
member. In Claude Code, start a fresh session and say: `Do slice S3 from docs/M2_SLICES.md.`
`CLAUDE.md` holds the workflow every slice follows.

Decisions already taken: Python **3.13** stays (our decision: the pinned numpy 2.5.4 and shap 0.54.1 need Python >= 3.12); deployment on
**Render** pulling a prebuilt image from **GHCR**; monitoring endpoint is **`/stats`**.

## Overview

| ID | Slice | Branch | Owner | Depends on | Priority |
|---|---|---|---|---|---|
| S0 | Project setup | `chore/project-setup` | Piranchal | none | M2 required |
| S1 | Production model in the repo | `feat/production-model` | Piranchal | S0 | M2 required |
| S2 | MLflow and reports hygiene | `infra/mlflow-hygiene` | Piranchal | S1 | M2 required |
| S3 | `/stats` monitoring | `feat/stats-endpoint` | Piranchal | S0 | M2 required |
| S4 | API hardening | `fix/api-hardening` | Piranchal | S3 | M2 required |
| S5 | Dockerfile | `infra/dockerfile` | Piranchal | S1, S4 | M2 required |
| S6 | CI: lint, test, build | `infra/ci-three-jobs` | Piranchal | S5 | M2 required |
| S7 | Render deployment | `infra/render-deploy` | Piranchal | S6 | M2 required |
| S8 | RUNBOOK | `docs/runbook` | Hussain | S7 | M2 required |
| S9 | Frozen evaluation set | `feat/frozen-eval-set` | Hussain | S0 | Proposal promise |
| S10 | Promotion gate | `feat/promotion-gate` | Hussain | S9 | Proposal promise |
| S11 | Retraining loop | `feat/retrain-loop` | Hussain | S2, S6, S10 | Proposal promise |
| S12 | Feature-set ablation | `feat/ablation` | Hussain | S2 | Nice to have |
| S13 | M2 documentation | `docs/m2-docs` | Hussain (both write their own reflections) | all | M2 required |

**Cut line if behind on Friday:** S12 moves to the final phase, and S11 ships as a dry run (gate
decision and report, no automatic PR). The course allows the focus area to be "in progress" at M2.

## Schedule (confirm the deadline on LMS)

Owners changed on 24 Nov: Piranchal owns S0 to S7 (the M2 baseline) and Hussain owns S8 to S13
(RUNBOOK, retraining loop, ablation, documentation). Reviews do not need the shared laptop: the
reviewer approves on github.com from any device. Because the split follows the pipeline, each member
must walk the other through their slices before the viva.

| Day | Piranchal | Hussain |
|---|---|---|
| Tue 24 | S0 (Hussain approves remotely as soon as the PR is open), then S1 and S3 | Reviews; S9 when available |
| Wed 25 | S2 (after S1 merges), S4 (after S3 merges) | S9 |
| Thu 26 | S5 | S10 |
| Fri 27 | S6 | S11 (the PR flow needs S6 merged) |
| Sat 28 | S7 | S11 finished and first retrain round; S8 (after S7 merges); S12 |
| Sun 29 | Clean-clone dry run; record the video's baseline parts (local run, Docker, `/stats`, CI, deployment) | S13; record the Project Focus and next-steps part |
| Deadline | Tag `v2.0-milestone2`, confirm CI green on the tag, submit | |

## Finding IDs used below

From the Milestone 1 review. A = modelling, B = plumbing, C = API, D = process/docs.
A2 confounded model comparison, A3 dyre explanation, A5 Tranco label noise, A6 selection on the
test set, B1 model only on one laptop, B2 committed MLflow store mutates, B3 promotion compares
stored numbers, B4 reports not tied to the deployed model, B6 slow gradient boosting, B7 mixed
requirements, C1 private suffixes, C2 punycode, C3 `/health` ignores explainer, C4 no raw length
limit, C5 `/explain` units, C6 `/` returns 404, C7 no monitoring, D1 review-claim contradiction,
D3 undocumented proposal deviations, D4 UMUDGA citation venue, D6 group-of-two note.

---

## S0: Project setup

**Do**
- Commit `CLAUDE.md` and this file (`docs/M2_SLICES.md`).
- Add `.gitattributes`: `* text=auto`, plus `eol=lf` for `*.sh`, `Dockerfile`, `*.yml`, `*.yaml`
  (Windows line endings break shell scripts inside Linux containers).
- Add `.github/pull_request_template.md`: slice ID, summary, acceptance checklist with evidence,
  out-of-scope findings, `Closes #`.
- Create one GitHub issue per slice S1 to S13 with `gh issue create`, using the slice text as the
  body, label `m2`, and assign the owner (`hmazhar008` or `Piranchall`).

**Manual (repo owner, GitHub web UI)**
- Settings, Rules, Rulesets: target `main`; require a pull request with 1 approval; dismiss stale
  approvals; require status check `test` (change to `lint`, `test`, `build` after S6); block force
  pushes; no bypass list.
- Make sure both members are collaborators with write access.

**Acceptance**
- Issues S1 to S13 exist and are assigned. A direct push to `main` is rejected.

## S1: Production model in the repo (B1, B7)

**Do**
- Add `models/production/` with the gb model Hussain trained for M1
  (`model_20261105_062411.joblib`, whose MLflow run `9c15d284722c46fc9355f98e134b44d1` is in the
  committed store; SHA-256
  `140989615550baed7e883885e0bfe3fd3b855b79e11bc70ff2d9f3d9f3a11b47`, verified on 24 Nov against the
  committed metrics) and a
  `latest.json` that records: path, version, model_type, `sha256`, `mlflow_run_id`,
  scikit-learn version, Python version, feature names, and `phased_in_families: []`.
- `.gitignore`: keep ignoring `models/*` but allow `models/production/`.
- `model_loader`: the default pointer becomes `models/production/latest.json`; the `MODEL_POINTER`
  env var still overrides it (document `MODEL_POINTER=models/latest.json` for serving a locally
  trained candidate). Verify the SHA-256 at load and refuse a mismatching file (API returns 503 and
  logs the reason). Warn if the running scikit-learn version differs from the recorded one.
- Split requirements: `requirements-api.txt` holds only what `src.app` imports at runtime (check
  the imports; mlflow, pytest and ruff must not be in it). `requirements.txt` starts with
  `-r requirements-api.txt` and adds training and dev tools, so existing install commands still work.

**Acceptance**
- On a fresh clone, `pytest -q` runs the `sanity` tests (they are no longer skipped) and they pass,
  locally and in CI.
- `python -m uvicorn src.app:app` serves predictions without training first.
- In a clean venv, `pip install -r requirements-api.txt` then `python -c "import src.app"` works.
- A test proves a tampered model file is refused.

## S2: MLflow and reports hygiene (B2, B4)

**Do**
- Make the tracking store configurable: env vars for the tracking URI and artifact directory, with
  a default **gitignored** local store (for example `mlflow_local/`) for all new runs.
- Treat the committed M1 store (`mlflow.db`, `mlruns/`) as a read-only snapshot. Add one documented
  command that copies it to a gitignored working folder, runs `relocate_mlflow_store()` on the copy
  and prints the `mlflow ui` command. Nothing ever writes to the committed files again.
- `src.train` no longer rewrites `reports/` by default; `src.report` builds the reports for the
  model in the production pointer (looked up by its `mlflow_run_id`), plus the M1 comparison of the
  three parent runs.
- Update README commands.

**Acceptance**
- On a fresh clone: `prepare_data`, `train --model logreg`, `pytest`, `src.report` and the
  M1-viewer command leave `git status` clean.
- `reports/` regenerated from the M1 snapshot is identical to the committed version.
- A test checks the committed `mlflow.db` hash is unchanged after the viewer command.

## S3: `/stats` monitoring (C7)

**Do**
- Middleware that records, thread-safely: request count per endpoint and status class, average
  latency (ms) overall and per endpoint, uptime and start time.
- Prediction statistics from `/predict`: count per label and mean P(dga).
- `GET /stats` returns these as JSON together with `model_version`.
- One structured JSON log line per request: timestamp, method, path, status, latency_ms, and label
  and probability for predictions. Do not log raw request bodies.
- Document in the README that counters are in memory and reset on restart (Render sleeps idle
  free-tier services).

**Acceptance**
- Tests: counts rise with requests, average latency is positive, label counts match the
  predictions made, `/stats` works with no model loaded.
- README endpoint table updated.

## S4: API hardening (C1 to C6)

**Do** (all response changes must be additive; existing fields keep their meaning)
- C1, C2: add `in_scope` (bool) and `warning` (string or null) to `/predict` and `/explain`. The
  domain is out of scope when it sits under a private public-suffix entry (detect with
  `tldextract` using private domains, for example `x.blogspot.com`, `x.duckdns.org`) or contains
  an `xn--` label. Still return the prediction. Do **not** change `features.py`.
- C3: `/health` reports model and explainer status separately; an explainer failure shows as
  degraded.
- C4: limit the raw input length in the Pydantic model before normalisation.
- C5: `/explain` returns `output_units` (`log_odds` for gb and logreg, `probability` for rf).
- C6: `GET /` redirects to `/docs`.

**Acceptance**
- A test per item. `abcxyzqwrtplk.blogspot.com` returns `in_scope: false` with a warning.
- README and the OpenAPI examples updated.

## S5: Dockerfile

**Do**
- Multi-stage build on `python:3.13-slim` (as promised in the proposal): the builder installs
  `requirements-api.txt` into a venv; the runtime stage copies the venv, `src/`,
  `data/reference/` (features.py loads the word list at import) and `models/production/`, and
  runs as a non-root user.
- `EXPOSE 8000`; start uvicorn on `0.0.0.0` and `${PORT:-8000}` (Render sets `PORT`).
- `HEALTHCHECK` hitting `/health` with Python's `urllib` (slim images have no curl).
- `.dockerignore` excluding `venv/`, `.git/`, `data/raw/`, `data/processed/`, `mlruns/`,
  `mlflow.db`, `mlflow_local/`, `tests/`, caches.
- Build arg `GIT_SHA` exposed as an env var and returned by `/health`, so a deployment can be
  identified.
- README: one documented command to build and run; Makefile targets `docker-build`, `docker-run`.

**Acceptance**
- `docker ps` shows `(healthy)` within about 30 seconds of start.
- `/predict`, `/explain`, `/stats` work against the container.
- `pip list` inside the image shows no mlflow, pytest or ruff. Image size noted in the PR.

## S6: CI with lint, test and build jobs

**Do**
- `.github/workflows/ci.yml` with three jobs: `lint` (ruff check and format check), `test`
  (pytest), `build` (`needs: [lint, test]`). Triggers: pushes to `main`, pull requests, and `v*`
  tags. Cache pip.
- `build`: build the image with `GIT_SHA`, run it, wait until Docker reports it healthy, call
  `/predict` once, then stop it.
- On pushes to `main` and on `v*` tags only: log in to GHCR with `GITHUB_TOKEN`
  (`permissions: packages: write`) and push `ghcr.io/piranchall/dgalert` tagged `sha-<short sha>`,
  plus `main` or the version tag.

**Manual**
- After merge: make the GHCR package public (or give Render registry credentials), and change the
  ruleset's required checks to `lint`, `test`, `build`.

**Acceptance**
- A PR shows three separate green checks. A push to `main` publishes the image tags in GHCR.
- One deliberately failing test, on a throwaway branch, turns the run red (useful for the video).

## S7: Render deployment

**Manual (do first)**
- Create a Render web service from the existing image `ghcr.io/piranchall/dgalert:main`, free
  instance, health-check path `/health`.
- Copy the service's deploy hook URL into a GitHub Actions secret `RENDER_DEPLOY_HOOK`.

**Do**
- Add a `deploy` job to `ci.yml` (`needs: build`, only on pushes to `main`) that calls the deploy
  hook, then polls the live `/health` until it reports the new `GIT_SHA` (fail the job after a
  timeout). Check Render's current documentation for deploying a specific image tag through the
  hook.
- README "Deployment" section: live URL, how deployment works, and cURL examples against the live
  service for `/health`, `/predict`, `/explain` and `/stats` (note that PowerShell needs
  `curl.exe`).

**Acceptance**
- Merging a trivial change to `main` updates the live service's `GIT_SHA` with no manual step.

## S8: RUNBOOK

**Do**
- `RUNBOOK.md`: normal redeploy (automatic on merge; manual through the hook or dashboard); rollback
  to a previous image tag (find the tag in GHCR, deploy it on Render, confirm the `GIT_SHA` in
  `/health`); rolling back a model (revert the production-model PR); running a production image
  locally with `docker pull`; cold starts on the free tier; `/stats` resetting; what to check when
  the health check fails.
- Rehearse one real rollback and roll-forward, and record the tags and times in the RUNBOOK.

**Acceptance**
- The other member follows the RUNBOOK unaided and completes a rollback and a roll-forward.

## S9: Frozen evaluation set and validation split (A6, groundwork for B3)

**Do**
- Without changing any M1 output, derive in `prepare_data` (or a new `src/prepare_retrain.py`):
  - for each held-out family, a fixed 20% slice grouped by second-level label, the **frozen eval
    slice**; the other 80% is the **phase-in pool**;
  - the **frozen eval set** = `test_seen` plus all frozen eval slices;
  - a **validation split**: 10% of `train`, grouped by second-level label and stratified by
    family, used for model selection in retraining.
- A content hash of the frozen eval set (SHA-256 over sorted `domain,label` pairs) written to a
  committed `reports/eval_frozen.json` with its row counts per family.
- Phase-in batches in `config.py`: batch 1 is `matsnu`; propose the later batches in the plan, with
  the reasoning.

**Acceptance**
- M1 split counts unchanged. The hash is identical across two runs. No second-level label is
  shared between train, validation, phase-in pool and frozen eval set. Tests for all of this.

## S10: Promotion gate (B3)

**Do**
- `src/gate.py`: given champion, challenger, the frozen eval set and the newly added families,
  score **both** models on the same rows (never compare stored numbers) and return a decision with
  reasons. Promote only if all hold, with thresholds declared in `config.py`:
  - recall on the newly added families improves by at least 0.02;
  - F1 on the seen families drops by at most 0.01;
  - false-positive rate rises by at most 0.005;
  - the sanity domains still classify correctly;
  - the frozen eval set hash matches `reports/eval_frozen.json`.
- Write the decision and side-by-side metrics (overall and per family) as JSON.
- Leave `promote_if_better` in `train.py` for M1 reproduction; document that retraining uses the
  gate.

**Acceptance**
- Tests with small synthetic models for promote, and for reject on each criterion separately.

## S11: Retraining loop (B6, proposal KaizenML)

**Do**
- `src/retrain.py`: read the production pointer (round and `phased_in_families`), take the next
  batch from the phase-in pool, train `HistGradientBoostingClassifier` with fixed parameters
  (choose them once on the validation split and record why) on train minus validation plus all
  phased-in pool rows, log the run to MLflow (experiment `dgalert-m2-kaizen`), run the gate against
  the champion, and on promotion write the new model and `latest.json` into `models/production/`
  plus `reports/retrain/round_<n>.json`.
- `.github/workflows/retrain.yml`: `workflow_dispatch` (input `dry_run`) and a weekly schedule.
  Steps: checkout, Python 3.13, install, `prepare_data`, `retrain`, upload the MLflow store as a
  workflow artifact, and on promotion open a PR from `retrain/round-<n>` with the gate report as
  the body, requesting review from the other member.
- Token: PRs opened with the default `GITHUB_TOKEN` do not trigger other workflows, so CI would
  never run on them and the ruleset would block the merge. Use a fine-grained personal access token
  limited to this repo (contents and pull requests: write) stored as secret `RETRAIN_PR_TOKEN`.

**Acceptance**
- A manual run on `main` ends in either a PR (promotion) or a logged rejection with reasons. A
  rejection is a valid demo: the gate is doing its job.
- Merging the PR runs CI, publishes a new image and the live `/health` shows the new model version.

## S12: Feature-set ablation (A2)

**Do**
- `scripts/ablation.py`: logreg on all 15 features and a tree model on the 8 basic features, logged
  to MLflow, with results in `reports/ablation.json` (overall and per family). Expected from the
  review: logreg with 15 features gives f1_seen of about 0.829; trees on 8 basic features catch
  about 89% of dyre against about 61% with all 15.

**Acceptance**
- Deterministic results; README section on model comparison updated in S13.

## S13: Milestone 2 documentation (D1, D3, D4, D6, A3, A5)

**Do**
- README:
  - **Project Focus** section: what (KaizenML retraining against drift, with explainability as the
    secondary focus), why it fits, what is built and compared, how it will be demonstrated.
  - Deployment, monitoring and retraining sections (if not already added by S7 and S11).
  - **Changes since the proposal**: Python 3.13 instead of 3.11 (our decision: the pinned numpy 2.5.4 and shap
    0.54.1 need Python >= 3.12);
    committed sample instead of downloading in `prepare_data`; chrmor dataset not used; SQLite
    MLflow store; `.joblib`; `/stats` instead of `/metrics`; fresh Tranco per retrain dropped; gate
    redesigned, with the evidence.
  - Corrections: dyre wording (A3), feature-set confound with the S12 numbers (A2), Tranco label
    noise (A5), a Mac/Linux quick start that does not depend on Windows paths.
- `AI_USAGE.md`: correct the review claim (D1) and add M2 tool use.
- One consistent UMUDGA citation in README and slides (D4).
- `CONTRIBUTION.md`: Milestone 2 member-to-task table built from the merged PRs; group-of-two
  approval note (D6). Each member writes their own reflection.

**Acceptance**
- The other member follows the README from a clean clone to a running container and a live call
  without help.

---

## After Milestone 2 (final presentation)

| ID | Slice | Notes |
|---|---|---|
| F1 | Hyphen-insensitive features | Compute features on the label with hyphens removed and drop `hyphen_count`; the review measured evasion 0% to 98% detected and matsnu recall 0.05 to 0.26. Keep `scripts/hyphen_evasion.py` as a regression test in the gate. |
| F2 | Threshold tuning | Pick the threshold on the validation split for a target false-positive rate; report ROC and PR curves. |
| F3 | Drift signals | Flag rate and score distribution against a reference in `/stats`, with an alert. |
| F4 | Private suffixes, properly | Decide whether to score the label below a private suffix; retrain if features change. |
| F5 | All phase-in rounds | Per-family recall chart across rounds for the presentation. |
| F6 | Confidence intervals | Bootstrap intervals for the headline metrics. |
| F7 | Feature schema guard | Hash of feature names, word list and `features.py` stored with the data and model, checked at load. |
| F8 | Label noise | Experiment with filtering machine-generated-looking Tranco tail domains. |