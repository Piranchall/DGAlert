# Contribution

## Team

| Name | ERP ID | GitHub |
|---|---|---|
| Piranchal Mukesh | 29050 | Piranchall |
| Muhammad Hussain | 28985 | hmazhar008 |

## Milestone 1 tasks

Built from the git history of the pull requests merged into `main`. "Author" is the commit author
of every commit on that branch. All commits also carry a `Co-Authored-By: Claude` trailer (see
[AI_USAGE.md](AI_USAGE.md)).

| Task | PR | Branch | Author | Merged | Main changes |
|---|---|---|---|---|---|
| Scaffold | [#1](https://github.com/Piranchall/DGAlert/pull/1) | `infra/m1-scaffold` | Muhammad Hussain | 2026-11-04 | pinned `requirements.txt`, ruff/pytest config, Makefile, GitHub Actions CI, `src/` package, pathlib `config.py`, config tests |
| Features | [#2](https://github.com/Piranchall/DGAlert/pull/2) | `feat/features` | Muhammad Hussain | 2026-11-04 | SCOWL English word list and its build script, domain normalization, second-level label extraction, the 15 features in `src/features.py`, feature tests |
| Sample + prepare_data | [#3](https://github.com/Piranchall/DGAlert/pull/3) | `feat/prepare-data` | Muhammad Hussain | 2026-11-05 | Tranco list ID, excluded and held-out families, `scripts/make_sample.py` and the committed sample, `src/prepare_data.py` validation and grouped split, tests |
| Training + MLflow + reports | [#4](https://github.com/Piranchall/DGAlert/pull/4) | `feat/train` | Muhammad Hussain | 2026-11-05 | `src/train.py` (logreg, rf, grid-searched gb), MLflow tracking and the committed run store, `latest.json` promotion, `src/report.py` and `reports/`, tests |
| API core | [#6](https://github.com/Piranchall/DGAlert/pull/6) | `feat/api` | Piranchal Mukesh | 2026-11-05 | `src/model_loader.py`, FastAPI `/predict` and `/health` with DNS input validation, test fixtures, loader/endpoint/sanity tests |
| `/explain` | [#7](https://github.com/Piranchall/DGAlert/pull/7) | `feat/explain` | Piranchal Mukesh | 2026-11-05 | `src/explain.py` (SHAP for gb/rf, coefficient fallback for logreg), `POST /explain`, explanation tests |
| README | [#8](https://github.com/Piranchall/DGAlert/pull/8) | `docs/readme` | Piranchal Mukesh | 2026-11-05 | `README.md` |

PR #5 (`feat/api`) was closed without merging and reopened as #6, with the same code.

## Reflections

### Muhammad Hussain
I set up the repository and built the feature extraction, data sampling and preparation pipeline, and
the training script with MLflow tracking. The biggest lesson was that data decisions mattered more
than model choice: grouping the split by second-level label, holding out whole families, and
excluding the dynamic-DNS families shaped the results far more than tuning did. I also learned that
F1 can't be compared across test sets with different DGA shares, which changed how we report drift.
Next time I would version the test set with a hash from the start, so model comparisons are
guaranteed to be on the same data.

### Piranchal Mukesh
I built the API: the model loader, /predict and /health with input validation, the SHAP-based
/explain endpoint, and the README. I learned how important it is that serving uses exactly the same
feature pipeline as training, and how much clear 422 and 503 responses help when something goes
wrong. The hardest part was handling the different SHAP output formats across model types while
keeping /explain consistent with /predict. Seeing /explain show the hyphen as the main reason matsnu
domains are missed was the most useful moment of the milestone.
