# Contribution

## Team

| Name | ERP ID | GitHub |
|---|---|---|
| Piranchal Mukesh | 29050 | Piranchall |
| Muhammad Hussain | 28985 | hmazhar008 |

## Milestone 1 tasks

Built from the git history of the pull requests merged into `main`. "Author" is the commit author
of the commits on that branch. "Reviewed by" is the member who merged the pull request on GitHub;
no approvals were recorded through GitHub's review feature.

Up to PR #12, 30 of the 35 non-merge commits on `main` carry a `Co-Authored-By: Claude` trailer (see
[AI_USAGE.md](AI_USAGE.md)). The five without it are the repository's initial commit, an
`AI_USAGE.md` wording edit made on GitHub (`8366b23`), and the three commits of PR #11.

| Task | PR | Branch | Author | Reviewed by | Merged | Main changes |
|---|---|---|---|---|---|---|
| Scaffold | [#1](https://github.com/Piranchall/DGAlert/pull/1) | `infra/m1-scaffold` | Muhammad Hussain | Muhammad Hussain | 2026-11-04 | pinned `requirements.txt`, ruff/pytest config, Makefile, GitHub Actions CI, `src/` package, pathlib `config.py`, config tests |
| Features | [#2](https://github.com/Piranchall/DGAlert/pull/2) | `feat/features` | Muhammad Hussain | Muhammad Hussain | 2026-11-04 | SCOWL English word list and its build script, domain normalization, second-level label extraction, the 15 features in `src/features.py`, feature tests |
| Sample + prepare_data | [#3](https://github.com/Piranchall/DGAlert/pull/3) | `feat/prepare-data` | Muhammad Hussain | Muhammad Hussain | 2026-11-05 | Tranco list ID, excluded and held-out families, `scripts/make_sample.py` and the committed sample, `src/prepare_data.py` validation and grouped split, tests |
| Training + MLflow + reports | [#4](https://github.com/Piranchall/DGAlert/pull/4) | `feat/train` | Muhammad Hussain | Muhammad Hussain | 2026-11-05 | `src/train.py` (logreg, rf, grid-searched gb), MLflow tracking and the committed run store, `latest.json` promotion, `src/report.py` and `reports/`, tests |
| API core | [#6](https://github.com/Piranchall/DGAlert/pull/6) | `feat/api` | Piranchal Mukesh | Piranchal Mukesh | 2026-11-05 | `src/model_loader.py`, FastAPI `/predict` and `/health` with DNS input validation, test fixtures, loader/endpoint/sanity tests |
| `/explain` | [#7](https://github.com/Piranchall/DGAlert/pull/7) | `feat/explain` | Piranchal Mukesh | Muhammad Hussain | 2026-11-05 | `src/explain.py` (SHAP for gb/rf, coefficient fallback for logreg), `POST /explain`, explanation tests |
| README | [#8](https://github.com/Piranchall/DGAlert/pull/8) | `docs/readme` | Piranchal Mukesh | Muhammad Hussain | 2026-11-05 | `README.md` |
| Contribution + AI usage | [#9](https://github.com/Piranchall/DGAlert/pull/9) | `docs/contribution` | Muhammad Hussain | Piranchal Mukesh | 2026-11-05 | `CONTRIBUTION.md`, `AI_USAGE.md` |
| Portable MLflow store | [#11](https://github.com/Piranchall/DGAlert/pull/11) | `fix/mlflow-portable-store` | Muhammad Hussain | Muhammad Hussain | 2026-11-05 | `src/mlflow_store.py` rewrites the committed store's absolute artifact paths for the current clone, called from `train.py` and `report.py`; tests; README note |
| Domain validation + demo domains | [#12](https://github.com/Piranchall/DGAlert/pull/12) | `fix/domain-validation` | Piranchal Mukesh | Muhammad Hussain | 2026-11-05 | reject domains with no known public suffix (IPs, internal names); demo examples switched to unseen `test_seen` domains; README |
| Limitations + this table | [#13](https://github.com/Piranchall/DGAlert/pull/13) | `docs/limitations` | Piranchal Mukesh | Muhammad Hussain | 2026-11-05 | `scripts/hyphen_evasion.py`, README limitations (hyphen evasion, false alarms), tuning caveats, this table |

PR #5 (`feat/api`) was closed without merging and reopened as #6, with the same code. Row #13 was
added in Milestone 2: that PR wrote this table and was merged before the `v1.0-milestone1` tag, but
was missing from it.

## Milestone 2 tasks

Built from the pull requests merged into `main` for the slices in `docs/M2_SLICES.md`.

- **Author:** the commit author of the commits on the branch.
- **Approved by:** whoever approved the PR through GitHub's review feature.
- **Merged by:** the account that merged it.

| Slice | PR | Branch | Author | Approved by | Merged by | Merged | Main changes |
|---|---|---|---|---|---|---|---|
| S0 Project setup | [#28](https://github.com/Piranchall/DGAlert/pull/28) | `chore/project-setup` | Piranchal Mukesh | none | Muhammad Hussain | 2026-11-24 | `CLAUDE.md`, `docs/M2_SLICES.md` |
| S0 Project setup (remaining files) | [#29](https://github.com/Piranchall/DGAlert/pull/29) | `chore/project-setup-2` | Piranchal Mukesh | Muhammad Hussain | Piranchal Mukesh | 2026-11-24 | `.gitattributes`, pull request template |
| S1 Production model in the repo | [#30](https://github.com/Piranchall/DGAlert/pull/30) | `feat/production-model` | Piranchal Mukesh | Muhammad Hussain | Piranchal Mukesh | 2026-11-24 | `models/production/` with the M1 gb model, SHA-256 check at load, `requirements-api.txt` |
| S3 `/stats` monitoring | [#31](https://github.com/Piranchall/DGAlert/pull/31) | `feat/stats-endpoint` | Piranchal Mukesh | Muhammad Hussain | Piranchal Mukesh | 2026-11-24 | `GET /stats`, request middleware, JSON access log |
| S2 MLflow and reports hygiene | [#32](https://github.com/Piranchall/DGAlert/pull/32) | `infra/mlflow-hygiene` | Piranchal Mukesh | Muhammad Hussain | Piranchal Mukesh | 2026-11-24 | read-only M1 snapshot, local MLflow store, `src.m1_snapshot`, reports tied to the production model |
| S4 API hardening | [#33](https://github.com/Piranchall/DGAlert/pull/33) | `fix/api-hardening` | Piranchal Mukesh | none | Piranchal Mukesh | 2026-11-25 | scope warnings (private suffixes, punycode), `degraded` health, input limit, `output_units`, `/` redirect |
| S5 Dockerfile | [#34](https://github.com/Piranchall/DGAlert/pull/34) | `infra/dockerfile` | Piranchal Mukesh | none | Piranchal Mukesh | 2026-11-25 | multi-stage non-root image with a health check, `GIT_SHA` in `/health` |
| S6 CI: lint, test, build | [#35](https://github.com/Piranchall/DGAlert/pull/35) | `infra/ci-three-jobs` | Piranchal Mukesh | Muhammad Hussain | Piranchal Mukesh | 2026-11-25 | three CI jobs, smoke-tested image published to GHCR |
| S7 Render deployment | [#37](https://github.com/Piranchall/DGAlert/pull/37) | `infra/render-deploy` | Piranchal Mukesh | Muhammad Hussain | Muhammad Hussain | 2026-11-25 | deploy job calling the Render deploy hook with the tested image, README Deployment |
| S8 RUNBOOK | [#38](https://github.com/Piranchall/DGAlert/pull/38) | `docs/runbook` | Muhammad Hussain | Piranchal Mukesh | Piranchal Mukesh | 2026-11-25 | `RUNBOOK.md`, `deploy.yml` for any `sha-` tag (rollback and roll-forward) |
| S8 RUNBOOK rehearsal | [#39](https://github.com/Piranchall/DGAlert/pull/39) | `docs/runbook-rehearsal` | Muhammad Hussain | Piranchal Mukesh | Piranchal Mukesh | 2026-11-25 | rehearsal record by both members, fixes found while following the RUNBOOK |
| S9 Frozen evaluation set | [#40](https://github.com/Piranchall/DGAlert/pull/40) | `feat/frozen-eval-set` | Muhammad Hussain | Piranchal Mukesh | Muhammad Hussain | 2026-11-25 | `src/prepare_retrain.py`, validation split, phase-in pool, frozen eval hash |
| S10 Promotion gate | [#41](https://github.com/Piranchall/DGAlert/pull/41) | `feat/promotion-gate` | Muhammad Hussain | Piranchal Mukesh | Piranchal Mukesh | 2026-11-25 | `src/gate.py` with six criteria |
| S11 Retraining loop | [#42](https://github.com/Piranchall/DGAlert/pull/42) | `feat/retrain-loop` | Muhammad Hussain | Piranchal Mukesh | Piranchal Mukesh | 2026-11-25 | `src/retrain.py`, `retrain.yml`, fixed HGB parameters |
| S12 Feature-set ablation | [#43](https://github.com/Piranchall/DGAlert/pull/43) | `feat/ablation` | Muhammad Hussain | Piranchal Mukesh | Piranchal Mukesh | 2026-11-25 | `scripts/ablation.py`, `reports/ablation.json` |
| S13 M2 documentation | [#44](https://github.com/Piranchall/DGAlert/pull/44) | `docs/m2-docs` | Muhammad Hussain | Piranchal Mukesh | Piranchal Mukesh | 2026-11-25 | README (Project Focus, changes since the proposal, corrections), `AI_USAGE.md`, this table |

PR #36 (`demo/failing-test`, "DEMO: failing test (do not merge)") was opened to show that CI
blocks a failing test, and closed without merging.

The owners changed on 24 Nov: Piranchal owns S0 to S7, and Muhammad Hussain owns S8 to S13.

### Approvals in a group of two (D6)

**In a team of two, the only possible reviewer of a PR is the other member.** GitHub does not let
an author approve their own pull request. So the "1 approval" rule always means "approved by the
other member".

**What the records show:**

- **S0 (#28) was meant to set up a ruleset requiring that approval, but none was created then.**
  `main` also has no classic branch protection: `gh api repos/Piranchall/DGAlert/branches/main/protection`
  returns 404.
- **The approvals on #29 to #32 were voluntary,** and so was the one on #35 (S6), which was merged
  on 25 Nov at 05:31 UTC, also before the ruleset existed.
- **#33 (S4) and #34 (S5) were merged without waiting for a review.**
  - They were merged on 25 Nov at 04:16 and 04:41 UTC.
  - Neither has a review, a review comment or a conversation comment, before or after the merge.
- **The ruleset `protect-main` was created on 25 Nov at 09:15 UTC, once we noticed.**
  - It requires one approval, dismisses stale approvals on new pushes, requires the `lint`,
    `test` and `build` checks, and blocks force pushes and deletion of `main`.
  - It has no bypass list.
  - Every PR merged after it was created (#37 to #44) has the other member's approval.

In Milestone 1 no pull request was approved through GitHub's review feature (see the table above
and [AI_USAGE.md](AI_USAGE.md#our-responsibility)).

## Reflections

### Milestone 1

#### Muhammad Hussain
I set up the repository and built the feature extraction, data sampling and preparation pipeline, and
the training script with MLflow tracking. The biggest lesson was that data decisions mattered more
than model choice: grouping the split by second-level label, holding out whole families, and
excluding the dynamic-DNS families shaped the results far more than tuning did. I also learned that
F1 can't be compared across test sets with different DGA shares, which changed how we report drift.
Next time I would version the test set with a hash from the start, so model comparisons are
guaranteed to be on the same data.

#### Piranchal Mukesh
I built the API: the model loader, /predict and /health with input validation, the SHAP-based
/explain endpoint, and the README. I learned how important it is that serving uses exactly the same
feature pipeline as training, and how much clear 422 and 503 responses help when something goes
wrong. The hardest part was handling the different SHAP output formats across model types while
keeping /explain consistent with /predict. Seeing /explain show the hyphen as the main reason matsnu
domains are missed was the most useful moment of the milestone.

### Milestone 2

#### Muhammad Hussain
I built the RUNBOOK and the reusable deploy workflow we use for rollbacks, then the frozen evaluation set, the six-criterion promotion gate, the retraining loop and the feature-set ablation. Round 1 taught me the most. Adding matsnu raised its recall from 0.056 to 0.873, but the gate rejected the model because the false-positive rate went from 0.1066 to 0.1235; it had learned that hyphenated dictionary words mean DGA. The proposal’s rule that F1 must improve would have promoted that model, so I now think a gate has to check false positives as well as F1. Before phasing matsnu in again, I’d fix the hyphen features.

#### Piranchal Mukesh
I built the Milestone 2 baseline: the production model committed with a SHA-256 check, /stats and the API fixes, the Docker image, the three CI jobs that publish it to GHCR, and the Render deployment, which waits until the live /health reports the merged commit. The thing I’ll remember is that production should run the exact image CI tested, and that this has to be checked on every deploy, which is why each one is verified through git_sha. Our process needed the same checking. We thought main was protected when it wasn’t, and two PRs were merged without review before I set up the ruleset.
