# AI usage

We used two AI tools, both from Anthropic.

- **Claude Code** wrote code, tests and documentation in the repository, in both milestones.
- **Claude (chat, claude.ai)** helped plan both milestones, write the prompts and rules we gave
  Claude Code, and review its plans.

## Milestone 1

### Claude (chat): planning and design

We used Claude in a chat to:

- plan the milestone and split it into tasks, one branch and pull request per task
- talk through design decisions before writing any code:
  - features on the second-level label only
  - reference n-gram and dictionary data from a fixed word list, never from training data
  - held-out families as simulated future malware
  - a split grouped by second-level label
  - F1 on `test_seen` as the pre-declared primary metric
- write the Claude Code prompt for each task, including the acceptance checks (ruff, pytest, real outputs to show)

The decisions we settled on were written down as fixed project rules, and Claude Code had to
follow them in every task.

### Claude Code: implementation, per task

| Task | PR | What Claude Code did |
|---|---|---|
| Scaffold | #1 | Pinned `requirements.txt`, ruff/pytest config, Makefile, GitHub Actions CI, `src/` package layout, `config.py`, config tests |
| Features | #2 | Word-list build script, `normalize_domain`, label extraction, the 15 features, tests with hand-computed values |
| Sample + prepare_data | #3 | `make_sample.py`, `prepare_data.py` (validation, features, grouped split), tests on a synthetic sample |
| Training + MLflow + reports | #4 | `train.py` (three model types, grid search, MLflow logging, `latest.json` promotion), `report.py`, tests on a tiny fixture |
| API core | #6 | `model_loader.py`, FastAPI `/predict` and `/health` with input validation, test fixtures and endpoint tests |
| `/explain` | #7 | `explain.py` (SHAP for tree models, coefficient fallback for logreg), `POST /explain`, tests including SHAP additivity checks |
| README | #8 | `README.md`, with every number read from `reports/`, `SAMPLE_INFO.md` and `split_report.json`. It checked the prompt's claims against the data and corrected two before writing. |
| Contribution + AI usage | #9 | `CONTRIBUTION.md` (task table from the merged-PR history) and this file |

For every task Claude Code ran `ruff check`, `ruff format --check` and `pytest -q` and showed us the
output before committing. Where a task called for it, it also ran the real pipeline or the API and
showed the actual outputs.

Up to PR #12, 30 of the 35 non-merge commits on `main` carry a `Co-Authored-By: Claude` trailer
(details in [CONTRIBUTION.md](CONTRIBUTION.md)).

## Milestone 2

### Claude (chat): review, planning and setup

We used Claude in a chat (claude.ai) to:

- **review Milestone 1 critically.** The findings (A, B, C and D IDs, e.g. A3 for the dyre
  explanation, D1 for the review claim) are the ones `docs/M2_SLICES.md` refers to.
- **draft `CLAUDE.md` and `docs/M2_SLICES.md`,** the workflow rules and the slice plan below.
- **review every slice plan** Claude Code presented, before we approved or changed it.
- **walk us through the setup outside the repository:** the GitHub ruleset, publishing to GHCR,
  the Render service and its deploy hook, and the RUNBOOK rehearsal.

### Claude Code: implementation, per slice

**How we worked.** Milestone 2 was split into the slices S0 to S13 in `docs/M2_SLICES.md`, each
with an owner, an issue, a branch and a pull request. `CLAUDE.md`, committed in S0, sets the rules
Claude Code had to follow in every slice:

- **Identity:** check that the active git and GitHub identity is the slice owner's before
  starting.
- **Plan first:** present a plan and wait for the owner's approval before editing.
- **Scope:** stay inside the slice; list anything else as an out-of-scope finding.
- **Tests:** add a test for every behaviour change, needing no raw data or network.
- **Checks:** run `ruff check`, `ruff format --check` and the full `pytest` suite and report the
  results before handing over.
- **Hand-over:** write the PR description and hand over the git commands. **Claude Code never
  committed, pushed, opened a PR or merged:** the slice owner ran those commands, and the other
  member reviewed and merged.

**What Claude Code did.** It wrote the code, tests and documentation of every slice, and ran
the pipeline, the API, Docker and the retraining round locally, showing the outputs. The decisions
were the owner's:

- **The plans:** we approved or changed every plan, after reviewing it with Claude (chat).
  Examples:
  - the sixth gate criterion (recall on earlier phased-in families);
  - re-training all four ablation cells as a reproducibility check;
  - not reverting the production model's PR in a model rollback.
- **Steps outside the repository:** we did these ourselves:
  - creating the Render service, the ruleset and the `RETRAIN_PR_TOKEN` secret;
  - running the deploy and rollback rehearsal recorded in `RUNBOOK.md`;
  - starting the real retraining round.

**Analyses it ran on request.** Two findings in the README come from read-only analyses that Claude
Code ran when we asked:
- why round 1 raised the false-positive rate;
- the SHAP and feature-range check behind the corrected dyre explanation.

**How AI involvement is recorded, and where the record is incomplete.**

- **Commit trailers:** only 4 of the 50 non-merge commits added between `v1.0-milestone1` and the
  merge of S12 (#43) carry a `Co-Authored-By: Claude` trailer (the S12 commits). The commit
  commands Claude Code handed over for the earlier slices did not include the trailer. So, unlike
  Milestone 1, the trailers do not show which commits Claude Code helped write. **In practice it
  helped write every slice.**
- **PR descriptions:** those of #37 to #43 end with "Generated with Claude Code"; those of #28 to
  #35 do not, although Claude Code wrote them too.

## Our responsibility

- **Review.** The Milestone 1 version of this file said "both members reviewed every pull request
  before it was merged". The records do not support that:
  - no Milestone 1 pull request was approved through GitHub's review feature (0 reviews on all 11
    merged PRs);
  - 5 of them (#1 to #4, #6) were merged by the account that opened them.
  - Reviews in Milestone 1 were informal and are not recorded.
  - In Milestone 2, #33 and #34 were merged without a review. Since the ruleset was created on
    25 Nov, every pull request needs the other member's approval, and every one merged since has
    it. Details are in [CONTRIBUTION.md](CONTRIBUTION.md#approvals-in-a-group-of-two-d6).
- **Running the code:** we ran the pipeline, the training, the API, the deployment and the
  retraining workflow ourselves.
- **Understanding:** We have understood the codebase and are able to explain it.
- **Numbers:** all metrics and row counts in the documentation come from files the pipeline
  produced or from commands whose output we saw. None were written by hand or by an AI from
  memory.
