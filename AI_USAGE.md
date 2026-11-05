# AI usage

We used two AI tools in Milestone 1, both from Anthropic.

- **Claude Code** wrote code, tests and documentation in the repository.
- **Claude (chat)** helped plan the work and write the prompts we gave Claude Code.

Every commit Claude Code helped write carries a `Co-Authored-By: Claude` trailer, so the AI
contribution can be traced in the git history.

## Claude (chat): planning and design

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

## Claude Code: implementation, per task

| Task | PR | What Claude Code did |
|---|---|---|
| Scaffold | #1 | Pinned `requirements.txt`, ruff/pytest config, Makefile, GitHub Actions CI, `src/` package layout, `config.py`, config tests |
| Features | #2 | Word-list build script, `normalize_domain`, label extraction, the 15 features, tests with hand-computed values |
| Sample + prepare_data | #3 | `make_sample.py`, `prepare_data.py` (validation, features, grouped split), tests on a synthetic sample |
| Training + MLflow + reports | #4 | `train.py` (three model types, grid search, MLflow logging, `latest.json` promotion), `report.py`, tests on a tiny fixture |
| API core | #6 | `model_loader.py`, FastAPI `/predict` and `/health` with input validation, test fixtures and endpoint tests |
| `/explain` | #7 | `explain.py` (SHAP for tree models, coefficient fallback for logreg), `POST /explain`, tests including SHAP additivity checks |
| README | #8 | `README.md`, with every number read from `reports/`, `SAMPLE_INFO.md` and `split_report.json`. It checked the prompt's claims against the data and corrected two before writing. |
| Contribution + AI usage | this PR | `CONTRIBUTION.md` (task table from the merged-PR history) and this file |

For every task Claude Code ran `ruff check`, `ruff format --check` and `pytest -q` and showed us the
output before committing. Where a task called for it, it also ran the real pipeline or the API and
showed the actual outputs.

## Our responsibility

- **Review:** both members reviewed every pull request before it was merged.
- **Running the code:** we ran the pipeline, the training and the API ourselves.
- **Understanding:** both members can explain every line of the code.
- **Numbers:** all metrics and row counts in the documentation come from files the pipeline
  produced. None were written by hand or by an AI from memory.
