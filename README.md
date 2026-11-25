# DGAlert

**Flags malware-generated domain names (DGA) from the domain string alone, served as a FastAPI service.**

DGAlert classifies a domain such as `moiqidfvjjhjmndqp.in` as `benign` or `dga` (made by a malware
Domain Generation Algorithm) using 15 lexical features of its second-level label, and explains each
decision with SHAP. It is an MLOps course project (BS CS, IBA, Fall 2026). Milestone 1
(`v1.0-milestone1`) built the model and the API. Milestone 2 runs the API in Docker, tests it in
CI, deploys it to Render, monitors it through `/stats`, and adds a retraining loop with a promotion gate (KaizenML) for new
malware families.

Live service: **https://dgalert.onrender.com** (docs at `/docs`).

## Contents

1. [Project Focus](#project-focus)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Data](#data)
5. [Results](#results)
6. [Explainability and the drift story](#explainability-and-the-drift-story)
7. [API](#api)
8. [Docker](#docker)
9. [Deployment](#deployment)
10. [Limitations](#limitations)
11. [Changes since the proposal](#changes-since-the-proposal)
12. [Tests](#tests)
13. [Citations](#citations)

## Project Focus

**What.** The main focus is **KaizenML**: automated retraining against drift. New DGA families
keep appearing, so the model is retrained in rounds, and a retrained model replaces the
production model only after a promotion gate accepts it. The second focus is **explainability**:
`/explain` returns the SHAP contributions behind every prediction. That is how the main drift in
our data was found (matsnu, below).

**Why it fits.** A lexical DGA classifier ages. In Milestone 1, 14 of the 46 usable UMUDGA
families were held out as "future" malware, and the production model misses several of them:
`matsnu` recall 0.0494, `nymaim` 0.2107, `simda` 0.4315, against 0.8853 on the families it
trained on. No live DGA feed is freely available, so the held-out families stand in for new
malware.

**What is built and compared.**

- A **frozen evaluation set** with a committed hash
  ([Retraining data](#retraining-data-milestone-2)). Every candidate is scored on the same rows.
- A **promotion gate** that scores the production model (champion) and the retrained model
  (challenger) on those rows and checks six criteria: recall on the new families, F1 on the seen
  families, false-positive rate, recall on earlier phased-in families, sanity domains, and the
  frozen hash ([Promotion gate](#promotion-gate-milestone-2)).
- A **retraining loop** in GitHub Actions. Each round phases in the next batch of held-out
  families, retrains, logs to MLflow and runs the gate. A promotion opens a pull request that the
  other member reviews; merging it deploys the new model
  ([Retraining loop](#retraining-loop-milestone-2)).
- A **feature-set ablation**, which separates the model family from the feature set in the M1
  comparison ([Feature-set ablation](#feature-set-ablation-milestone-2)).

**What happened so far.** Round 1 (matsnu) ran on `main` (Actions run 45668715690). **The gate
rejected it.**

| Measure (frozen eval set) | Champion (gb, M1) | Challenger (hgb + matsnu) | Gate result |
|---|---|---|---|
| matsnu recall | 0.0560 | 0.8727 | pass |
| seen F1 | 0.8891 | 0.8850 | pass |
| false-positive rate | 0.1066 | 0.1235 | **fail** (+0.0169; at most +0.005 allowed) |
| sanity domains | 6/6 | 6/6 | pass |

The cause of the extra false positives was measured by also training the same model without the
matsnu rows:

- The model change (gb to HistGradientBoosting) adds +0.0031 to the false-positive rate.
- The matsnu rows add +0.0138: 396 benign domains are newly flagged.
- 57.8% of those newly flagged domains contain a hyphen, against 9.4% of all benign domains.
  Examples: `anime-planet.com`, `tiger-explorer.com`, `national-lottery.biz`.

matsnu joins two English words with a hyphen (`accident-club.com`). Before matsnu, no training
DGA label had a hyphen, so the model learns that a hyphen plus dictionary words means DGA, and
real hyphenated two-word sites look the same. Under the proposal's original rule (strict F1
improvement plus the sanity tests), this model would have been promoted, because F1 on the frozen
set rose from 0.8870 to 0.8952 (see
[Changes since the proposal](#changes-since-the-proposal)).

**How it is demonstrated.**

1. Actions → "Retrain" → Run workflow on `main`, with *dry_run* unticked.
2. The run's summary shows the gate report: each criterion with champion and challenger values,
   the decision and its reasons. The MLflow store and the report are kept as an artifact.
3. A rejection ends there, as in round 1. A promotion opens the PR
   "retrain: round <n> (automated)". Merging it deploys the new model, and the live `/health`
   then shows its `model_version`.

**Next steps (final presentation).**

- **Hyphen-insensitive features:** compute the features on the label with hyphens removed and drop
  `hyphen_count`. This addresses both the matsnu false positives and the hyphen evasion in
  [Limitations](#limitations).
- **Threshold tuning:** choose the decision threshold on the validation split for a target
  false-positive rate.
- **Drift signals in `/stats`:** compare the flag rate and the score distribution against a
  reference.

## Architecture

```mermaid
flowchart LR
    U["UMUDGA raw lists<br/>data/raw/umudga"] --> MS["scripts/make_sample.py"]
    T["Tranco top-1M (746WX)<br/>data/raw/tranco"] --> MS
    MS --> S[("data/sample/<br/>committed")]
    S --> P["src/prepare_data.py<br/>validate + features + grouped split"]
    P --> TR[("train.parquet")]
    P --> TS[("test_seen.parquet")]
    P --> TF[("test_future.parquet")]
    TR --> TN["src/train.py<br/>logreg · rf · gb (grid search)"]
    TS --> TN
    TF --> TN
    TN <--> ML[("MLflow local store<br/>mlflow_local/ (gitignored)")]
    M1[("M1 snapshot<br/>mlflow.db + mlruns/<br/>committed, read-only")] -. "copy (src.m1_snapshot)" .-> RP["src/report.py"]
    RP --> R[("reports/")]
    TN --> M["models/model_&lt;version&gt;.joblib<br/>+ models/latest.json<br/>(local candidate, gitignored)"]
    PR[("models/production/<br/>model + latest.json<br/>committed")] --> API["src/app.py (FastAPI)<br/>/predict · /explain · /health · /stats"]
    M -. "MODEL_POINTER=models/latest.json" .-> API
```

Milestone 2 adds the delivery pipeline and the retraining loop:

```mermaid
flowchart LR
    subgraph CI["GitHub Actions: ci.yml"]
        L["lint"] --> B["build + smoke test"]
        TE["test"] --> B
        B --> D["deploy.yml"]
    end
    MAIN["merge to main"] --> CI
    B -- "sha-&lt;short&gt;" --> G[("GHCR<br/>ghcr.io/piranchall/dgalert")]
    D -- "deploy hook + imgURL" --> R["Render<br/>dgalert.onrender.com"]
    G --> R
    subgraph RT["GitHub Actions: retrain.yml (manual; weekly dry run)"]
        PD["prepare_data +<br/>prepare_retrain"] --> RR["src/retrain.py<br/>next batch, fit hgb"]
        RR --> GT{"src/gate.py<br/>6 criteria on the<br/>frozen eval set"}
    end
    GT -- "promote" --> RPR["PR: retrain: round &lt;n&gt;<br/>(reviewed by the other member)"]
    GT -- "reject" --> REP["gate report +<br/>MLflow artifact"]
    RPR --> MAIN
```

All features are defined once in `src/features.py` and used by `prepare_data`, `train`, the API
and the explainer. Features use only the second-level label (`mail.google.co.uk` -> `google`,
via `tldextract` with its bundled public suffix list). The n-gram and dictionary features come from
a fixed English word list in `data/reference/`, never from the training data, to avoid leakage.

## Quick start

### From a clean clone to a running container and a live call

Needs only git and Docker (Docker Desktop on Windows/macOS), not Python. The same commands work
in bash and in PowerShell:

```bash
git clone https://github.com/Piranchall/DGAlert.git
cd DGAlert
docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t dgalert:local .
docker run --rm -d -p 8000:8000 --name dgalert dgalert:local
docker ps --filter name=dgalert      # STATUS shows "(healthy)" after about 10 seconds
```

Then call it, locally and on the live service. In bash:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"domain": "moiqidfvjjhjmndqp.in"}'
curl https://dgalert.onrender.com/health
curl -X POST https://dgalert.onrender.com/predict -H "Content-Type: application/json" -d '{"domain": "moiqidfvjjhjmndqp.in"}'
```

In Windows PowerShell 5.1, write `curl.exe` and escape the quotes inside the JSON body:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{\"domain\": \"moiqidfvjjhjmndqp.in\"}'
curl.exe https://dgalert.onrender.com/health
curl.exe -X POST https://dgalert.onrender.com/predict -H "Content-Type: application/json" -d '{\"domain\": \"moiqidfvjjhjmndqp.in\"}'
```

- **Local `/health`:** shows `"status": "healthy"` and, as `git_sha`, the commit you built.
- **`/predict`:** returns `"label": "dga"` for this domain.
- **The live service:** can take up to about a minute to answer the first call after it has been
  idle (see [Deployment](#deployment)).
- **Stop the container:** `docker stop dgalert`.
- **Interactive docs:** http://127.0.0.1:8000/docs.

### From source (Python)

**Requires Python 3.13.**

`data/raw/` is not needed: `prepare_data` reads the committed sample in `data/sample/`.

Windows (PowerShell, from the repo root):

```powershell
py -3.13 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt

venv\Scripts\python -m uvicorn src.app:app       # serves the committed model; no training needed
                                                 # API at http://127.0.0.1:8000, docs at /docs
venv\Scripts\python -m pytest -q

venv\Scripts\python -m src.prepare_data          # writes data/processed/
venv\Scripts\python -m src.prepare_retrain       # retraining data + reports/eval_frozen.json (after prepare_data)
venv\Scripts\python -m src.retrain --dry-run     # one retraining round, gate only (after prepare_retrain)
venv\Scripts\python -m src.train --model all     # trains logreg, rf, gb; writes models/latest.json
venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow_local/mlflow.db   # your runs, http://127.0.0.1:5000

venv\Scripts\python -m src.m1_snapshot             # copy of the M1 runs; prints its `mlflow ui` command
venv\Scripts\python -m src.report                  # rebuilds reports/ (needs data/processed/)
```

macOS / Linux (bash, from the repo root; `python3.13` must be on the PATH):

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -r requirements.txt

venv/bin/python -m uvicorn src.app:app          # serves the committed model; no training needed
                                                # API at http://127.0.0.1:8000, docs at /docs
venv/bin/python -m pytest -q

venv/bin/python -m src.prepare_data             # writes data/processed/
venv/bin/python -m src.prepare_retrain          # retraining data + reports/eval_frozen.json (after prepare_data)
venv/bin/python -m src.retrain --dry-run        # one retraining round, gate only (after prepare_retrain)
venv/bin/python -m src.train --model all        # trains logreg, rf, gb; writes models/latest.json
venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow_local/mlflow.db   # your runs, http://127.0.0.1:5000

venv/bin/python -m src.m1_snapshot              # copy of the M1 runs; prints its `mlflow ui` command
venv/bin/python -m src.report                   # rebuilds reports/ (needs data/processed/)
```

Training takes about 11 minutes. Most of it is the gradient boosting model: about 6 minutes of grid
search (27 grid points x 3 folds) and about 4.4 minutes refitting the best grid point on the full
train set.

The same steps as `make` targets on macOS / Linux (see `Makefile`):

| Step | Command |
|---|---|
| venv + install | `make setup` |
| prepare data | `make data` |
| retraining data (after `make data`) | `make retrain-data` |
| one retraining round, dry run (after `make retrain-data`) | `make retrain-dry-run` |
| train (all models) | `make train` |
| feature-set ablation, about 3 min (after `make data`) | `make ablation` |
| rebuild `reports/` | `make report` |
| browse the M1 runs (copy + `mlflow ui` command) | `make mlflow-m1` |
| tests | `make test` |
| ruff lint + format check | `make lint` |
| serve the API (with `--reload`) | `make serve` |

Training logs to a local, gitignored MLflow store in `mlflow_local/` and does not touch `reports/`.
`models/latest.json` is only replaced when the new best model has a strictly higher `f1_seen` than
the current one (`python -m src.train --force` overrides this). That rule is for reproducing
Milestone 1; retraining promotes through the [promotion gate](#promotion-gate-milestone-2).

### MLflow stores and reports

| Store | Where | Written by |
|---|---|---|
| Local store (new runs) | `mlflow_local/mlflow.db` + `mlflow_local/mlruns/`, gitignored | `src.train` |
| M1 snapshot | `mlflow.db` + `mlruns/`, committed | nothing: read-only |
| M1 working copy | `mlflow_m1/`, gitignored | `src.m1_snapshot`, `src.report` |

- **New runs** go to the local store. Set `DGALERT_MLFLOW_TRACKING_URI` (and, for a SQLite store,
  `DGALERT_MLFLOW_ARTIFACT_DIR`) to log elsewhere, e.g. a CI workspace. The names are prefixed so a
  global `MLFLOW_TRACKING_URI` set for another project cannot redirect DGAlert runs. Training
  refuses to write to the committed `mlflow.db`.
- **The M1 snapshot** (`mlflow.db`, `mlruns/`) holds the Milestone 1 runs and is never written
  again. Its SHA-256 is recorded in `src/config.py` (`M1_SNAPSHOT_DB_SHA256`) and the test suite
  fails if it changes. To browse it, make a working copy:

  ```powershell
  venv\Scripts\python -m src.m1_snapshot
  ```

  This checks the hash, copies the store to `mlflow_m1/` (replacing any earlier copy), rewrites the
  copy's artifact paths for this machine (MLflow stores them as absolute paths) and prints the
  `mlflow ui --backend-store-uri sqlite:///.../mlflow_m1/mlflow.db --port 5000` command to run.
- **`python -m src.report`** rebuilds `reports/` from a fresh working copy: `m1_results.json` and
  `per_family_recall.json` compare the three M1 parent runs, and `production_model.json` describes
  the model the API serves. It finds the MLflow run named by `mlflow_run_id` in
  `models/production/latest.json`, in the local store or the M1 snapshot, and records its params,
  metrics and per-family recall, plus the store's label (`local` or `m1_snapshot`). It
  needs `data/processed/split_report.json` (`python -m src.prepare_data`). The files contain no
  timestamps or paths, so rerunning it leaves `git status` clean.

### Production model and requirements

The API serves the **committed production model** in `models/production/` (the M1 `gb` model,
version `20261105_062411`, MLflow run `9c15d284722c46fc9355f98e134b44d1`). Its `latest.json`
records the file's SHA-256, the scikit-learn and Python versions, the feature names and the
families phased in by retraining (none yet). Training never writes there: `src.train` writes a
local, gitignored candidate to `models/` and `models/latest.json`. To serve that candidate instead:

```powershell
$env:MODEL_POINTER = "models/latest.json"; venv\Scripts\python -m uvicorn src.app:app
```

A relative `MODEL_POINTER` and the `path` inside a pointer are resolved against the repo root, so
the service works from any working directory. At startup the loader checks the model file's
SHA-256 against the pointer and refuses a mismatching file (`/health` and `/predict` return 503
and the server log gives the reason). A pointer without `sha256` still loads, with a warning, and a
warning is logged if the installed scikit-learn differs from the one the model was saved with.

Requirements are split: `requirements-api.txt` holds only what the API needs at runtime (no
mlflow, pytest or ruff; meant for a slim deployment image), and `requirements.txt` includes it
and adds the training and dev tools, so `pip install -r requirements.txt` still installs
everything.

## Data

### Sources

- **DGA domains: UMUDGA** (Zago, Gil Pérez and Martínez Pérez, *Data in Brief* 30, 2020; full
  reference under [Citations](#citations)). Dataset on Mendeley Data,
  DOI [10.17632/y8ph45msv8.1](https://doi.org/10.17632/y8ph45msv8.1), MIT licence.
  For each family only `<family>/list/10000.txt` is used; the family name is the folder name.
- **Benign domains: Tranco** top-1M list
  [`746WX`](https://tranco-list.eu/list/746WX/1000000). Tranco averages several popularity
  rankings over 30 days, so it is much harder to manipulate than a single ranking.

The raw files are downloaded manually into `data/raw/` (gitignored). `scripts/make_sample.py`
builds the committed sample in `data/sample/`; `data/sample/SAMPLE_INFO.md` records every
parameter, the Tranco file's SHA-256 and all drop counts.

### Sample

| | Rows |
|---|---|
| Domains per UMUDGA family | up to 3000 (seed 42) |
| DGA rows, training families | 96000 |
| DGA rows, held-out families | 42000 |
| DGA rows, total | 138000 |
| Benign rows (Tranco) | 96000 |

Benign rows are sampled **evenly across the ten 100k-rank bands** of Tranco (9600 per band), not
taken from the top N. The top of the list is dominated by short, famous brand names, which are
easy to tell apart from random strings. A DNS resolver sees the whole long tail, so the benign set
should too; otherwise the model learns "short and famous = benign" and the test scores are
optimistic.

The benign count equals the number of training-family DGA rows (96000), so train is about 50/50
benign vs DGA. The per-family cap of 3000 keeps any single family from dominating.

Cleaning dropped 6 Tranco `_wildcard_` rows, 1447 punycode (`xn--`) domains, 11 bare public
suffixes, and 10 domains that appeared both in Tranco and in a DGA list (dropped from both sides).

### Excluded families (never used)

`corebot`, `symmi`, `kraken_v1`, `kraken_v2`, `legit`.

- `corebot`, `symmi`, `kraken_v1`, `kraken_v2` put the generated string in a dynamic-DNS
  subdomain (e.g. `x.ddns.net`), so their second-level label is just the provider's name. A
  second-level-label classifier cannot see them.
- `legit` is UMUDGA's own benign list. It overlaps Tranco heavily, and Tranco is our benign source.

### Held-out families (simulated future malware)

`matsnu`, `nymaim`, `banjori`, `sisron`, `simda`, `dyre`, `padcrypt`, `shiotob`, `necurs`,
`qakbot`, `tinba`, `murofet_v1`, `murofet_v2`, `murofet_v3`.

These 14 of the 46 usable families (about 30%, chosen by the team) never appear in training.
Their rows go only to `test_future`, which simulates malware that appears after the model was
trained. `f1_future` and per-family recall on `test_future` are the drift baseline that
Milestone 2's automated retraining has to improve on. The set mixes word-based (matsnu, nymaim),
pronounceable or seed-based (banjori, simda, sisron) and random-character families.

### Split

`src/prepare_data.py` validates the sample, computes features, and splits it:

- Training families + benign are split 80/20 with `GroupShuffleSplit` **grouped by second-level
  label** (seed 42), so the same label (e.g. `google` from `google.com` and `google.de`) can
  never be in both train and test.
- `test_seen` is that 20%.
- `test_future` is the held-out-family DGA rows plus the benign rows of `test_seen`. Held-out
  rows whose label also occurs in train are dropped (3 rows).

| Split | Rows | Benign | DGA | Benign fraction |
|---|---|---|---|---|
| train | 153627 | 76856 | 76771 | 0.5003 |
| test_seen | 38373 | 19144 | 19229 | 0.4989 |
| test_future | 61141 | 19144 | 41997 | 0.3131 |

Second-level labels shared between train and test_seen: **0**. Between train and test_future: **0**
(`data/processed/split_report.json`).

### Retraining data (Milestone 2)

`src/prepare_retrain.py` (run after `prepare_data`) derives the data for the retraining loop from
the three splits above, which it does not change:

| Part | Rows | What it is |
|---|---|---|
| `train_fit` | 138248 | train minus validation: what retraining fits on (plus phased-in rows) |
| `validation` | 15379 | about 10% of train, for choosing hyperparameters |
| `phase_in_pool` | 33570 | about 80% of each held-out family: added to training one batch per round |
| `eval_frozen` | 46800 | `test_seen` (38373) + about 20% of each held-out family (8427): the fixed set every candidate is scored on |

- **One rule for both selections.** A second-level label goes to the frozen eval slice (or to
  validation) when `sha256("<salt>:<label>")`, read as a fraction, is below 0.2 (or 0.1).
  - The salts are `frozen-eval` and `validation`.
  - Membership depends only on the label, so it does not change when other rows or families
    change, and every domain with the same label lands on the same side.
  - Actual shares: 19.0% to 21.0% per held-out family, and 9.1% to 11.4% per family of train.
- **No leakage.** No second-level label is shared between any two parts; the script checks this
  and stops if one is.
- **Frozen hash.** The frozen eval set's hash (SHA-256 over its sorted `domain,label` lines) and its
  rows per family are committed in `reports/eval_frozen.json` (`sha256` `78f48f6b…`). The promotion
  gate (slice S10) is to refuse to compare models on a set with another hash.
- **Phase-in batches** (`config.PHASE_IN_BATCHES`), ordered by M1 gb recall on test_future, worst
  first:

| Batch | Families | M1 gb recall | Pool rows | Frozen eval rows |
|---|---|---|---|---|
| 1 | matsnu | 0.049 | 2409 | 589 |
| 2 | nymaim, simda | 0.211, 0.432 | 2391, 2430 | 609, 569 |
| 3 | dyre, shiotob | 0.646, 0.740 | 2381, 2370 | 619, 630 |
| never (control) | banjori, sisron, padcrypt, necurs, qakbot, tinba, murofet_v1, murofet_v2, murofet_v3 | 0.98 to 1.00 | 21589 | 5411 |

The control families are never phased in. Their recall cannot rise by the gate's margin, and
keeping them unseen still measures how the model does on families it was never trained on.

### Promotion gate (Milestone 2)

`src/gate.py` decides whether a retrained model (the challenger) replaces the production model
(the champion). It scores **both** models on the same rows of the frozen eval set and never
compares stored numbers. The challenger is promoted only if all six criteria hold (thresholds in
`src/config.py`, boundaries inclusive):

| # | Criterion | Measured on | Rule |
|---|---|---|---|
| 1 | frozen eval hash | the whole frozen eval set | equals `sha256` in `reports/eval_frozen.json` |
| 2 | new-family recall | DGA rows of the families added this round, pooled | rises by at least 0.02 |
| 3 | seen F1 | the `test_seen` rows (training families + benign) | drops by at most 0.01 |
| 4 | false-positive rate | benign rows flagged as dga | rises by at most 0.005 |
| 5 | earlier-family recall | DGA rows of families phased in during earlier rounds, pooled | drops by at most 0.02; passes when there are none (round 1) |
| 6 | sanity domains | `config.SANITY_DOMAINS` (the same list as the sanity tests) | all classified correctly |

Criterion 5 is **added beyond the slice plan's five**. Families phased in earlier are held-out
families, so their rows are not among the seen-F1 rows at all: without it, losing a ~600-row
earlier family completely would leave seen F1 unchanged and move F1 over all ~46.8k frozen rows
only slightly.

All six are always evaluated, so a rejection lists every failed criterion. The decision (JSON) has
`promote`, `failed`, one entry per criterion (champion and challenger values, delta, threshold,
message), and both models' metrics side by side: overall precision, recall and F1, seen F1,
new-family and earlier-family recall, false-positive rate, recall per family, and the sanity
labels. `src/train.py`'s `promote_if_better` (higher stored `f1_seen`) stays for reproducing
Milestone 1 only.

### Retraining loop (Milestone 2)

`src/retrain.py` runs one round: it phases in the next batch of held-out families, retrains, and
lets the gate decide.

1. **Plan the round.** Read the production pointer (`models/production/latest.json`): its `round` (0
   for the M1 model) and `phased_in_families`. Take the next batch of `config.PHASE_IN_BATCHES`.
2. **Train.** Fit a `HistGradientBoostingClassifier` (model type `hgb`) on `train_fit` plus the
   phase-in pool rows of every phased-in family, earlier and new.
   - The parameters are fixed in `config.HGB_PARAMS`, so a round is deterministic and takes one
     fit (about 10 s).
   - They were chosen once by `scripts/select_hgb_params.py`: 16 grid points, each fit on
     `train_fit` and scored by F1 on `validation`, results in
     `reports/retrain/hgb_param_search.json`.
   - Of the three points within 0.001 of the best F1 (0.8902), the cheapest won: learning rate
     0.1, 200 iterations, 63 leaves, F1 0.8894.
3. **Gate.** Run the promotion gate against the production model on the frozen eval set.
4. **Log.** Record the run in MLflow (experiment `dgalert-m2-kaizen`): parameters, both models'
   gate metrics, and the decision as an artifact.
5. **Report.** Always write `reports/retrain/round_<n>.json` and `round_<n>.md` (the gate report).
6. **On promotion** (not with `--dry-run`):
   - save the new model in `models/production/` and delete the old model file;
   - write `latest.json` (with `round`, `phased_in_families` and the frozen set's hash);
   - write `reports/production_model.json`.
   - To roll back a promoted model, see [RUNBOOK.md](RUNBOOK.md#roll-back-a-model).

```powershell
venv\Scripts\python -m src.prepare_data
venv\Scripts\python -m src.prepare_retrain
venv\Scripts\python -m src.retrain --dry-run   # train and gate only; models/production/ unchanged
```

**In GitHub Actions** (`.github/workflows/retrain.yml`):
- **Real rounds are manual:** Actions → "Retrain" → Run workflow on `main`, with *dry_run*
  unticked. When the gate promotes, the run opens the PR **"retrain: round <n> (automated)"** from
  `retrain/round-<n>`, with the gate report as its body.
  - The PR is opened with `RETRAIN_PR_TOKEN`, a fine-grained token of one member (this repo only,
    Contents and Pull requests read/write), and review is requested from the other member.
  - A PR opened with the default `GITHUB_TOKEN` would not trigger CI, and the ruleset would block
    its merge.
  - Merging the PR runs CI, publishes a new image and deploys it, and `/health` then shows the new
    `model_version`.
- **The weekly run (Mondays 03:00 UTC) is always a dry run.**
- **What every run does:** regenerates the data from `data/sample`, writes the frozen eval report
  to a temporary file so the committed `reports/eval_frozen.json` stays the reference, and uploads
  the MLflow store and the round report as an artifact (kept 90 days).
- **No duplicates:** a run does nothing while the PR for its round is still open.

**A rejected batch is retried unchanged.** The data is fixed and training is deterministic, so
every later round with the same features and parameters gets the same decision.
- **Round 1 (matsnu) was rejected on the false-positive rate,** in the real round on `main`
  (Actions run 45668715690) and identically in a local dry run.
  - matsnu recall: 0.056 → 0.873.
  - Seen F1: −0.004.
  - False-positive rate: 0.1066 → 0.1235, a rise of +0.0169 against the 0.005 allowed. Why it
    rises is in [Project Focus](#project-focus).
- **The planned fix** for the final presentation is hyphen-insensitive features.

**Note:** once a retrained model is in production, its MLflow run lives in the workflow artifact,
not in `mlflow_local/`. `python -m src.report` then needs `DGALERT_MLFLOW_TRACKING_URI` pointing at
the downloaded artifact's `mlflow.db`.

## Results

Three MLflow parent runs in experiment `dgalert-m1` (seed 42, threshold 0.5 on P(dga), positive
class = dga). Numbers come from `reports/m1_results.json`, which `src/report.py` builds from MLflow.

| Model | Features | f1_seen | precision_seen | recall_seen | f1_future | recall_future | train_time_s |
|---|---|---|---|---|---|---|---|
| logreg | 8 basic (no dictionary / n-gram) | 0.7635 | 0.7721 | 0.7552 | 0.8414 | 0.8003 | 0.6825 |
| rf | all 15 | 0.884 | 0.8828 | 0.8852 | 0.8549 | 0.7867 | 10.487 |
| **gb** | all 15 | **0.8891** | **0.8929** | **0.8853** | **0.8584** | **0.7885** | 261.2648 |

- **logreg**: `StandardScaler` + `LogisticRegression` (C=1.0, `class_weight="balanced"`).
- **rf**: `RandomForestClassifier`, 100 trees, `class_weight="balanced"`.
- **gb**: `GradientBoostingClassifier`, tuned by grid search (27 grid points, 3-fold `GroupKFold` by
  second-level label, on a stratified 40000-row subsample of train). Every grid point is a nested
  MLflow child run. Best grid point: `n_estimators=200`, `max_depth=7`, `learning_rate=0.05`, with
  CV F1 0.8841 ± 0.0026. It was then refit on the full train set.

**This table mixes two effects.** logreg got only the 8 basic features and the trees got all 15,
so the 0.1256 `f1_seen` gap between logreg and gb is partly the feature set. The
[feature-set ablation](#feature-set-ablation-milestone-2) separates the two.

### Why f1_seen and f1_future are not comparable

The two test sets have different class balance:

| Test set | Rows | DGA rows | DGA share |
|---|---|---|---|
| test_seen | 38373 | 19229 | 0.5011 |
| test_future | 61141 | 41997 | 0.6869 |

F1 depends on precision, and precision rises when positives are more common. With 68.69% DGA rows
in `test_future`, every model's precision there is inflated. That is why `f1_future` is *higher*
than `f1_seen` for logreg even though it misses whole families. **Compare recall instead:** gb
catches 0.8853 of DGA domains from families it has seen but only 0.7885 from families it has not.

### Why gb

`gb` is the deployed model (`models/production/`), chosen because:

1. **It is best on the metric we declared in advance, `f1_seen`** (0.8891 vs 0.884 for rf).
2. **Its artifact is much smaller** than the 100-tree, full-depth random forest. That matters for
   loading time now and for the Docker image in Milestone 2.
3. **SHAP is much faster on it.** `TreeExplainer` on shallow boosted trees (`max_depth=7`) is far
   cheaper than on fully grown random-forest trees, and `/explain` runs SHAP on every request.

To be honest about the margin: gb's lead over rf is small (0.0051 F1 on `test_seen`, 0.0035 on
`test_future`), and rf is better on some families (e.g. `suppobox_2`, `vawtrak_v3`, `rovnix` in
`test_seen`). On accuracy alone the two are close; points 2 and 3 decide it.

Two caveats about the tuning:

- **The best `max_depth` (7) is the largest value in the grid** (3, 5, 7), and the best
  `learning_rate` (0.05) is the smallest (0.05, 0.1, 0.2). So the search did not show that deeper
  trees or a smaller learning rate would be worse; it only found the best point inside this grid.
- **gb's `train_time_s` (261.2648) is the final refit only.** The grid search before it is not
  included. By the MLflow run timestamps it ran for 357 s of the 622 s gb run, which makes it the
  biggest single part of the ~11-minute training.

### Feature-set ablation (Milestone 2)

The table above compares logreg on 8 basic features with trees on all 15, so it mixes the model
family with the feature set. `python -m scripts.ablation` (`make ablation`) fits the full 2 x 2
grid on the same M1 splits: logreg with the M1 parameters, gb with M1's best grid point, each on the
8 basic and on all 15 features. Each cell is an MLflow run in experiment `dgalert-m2-ablation`
(local store); the results, overall and per family, are in `reports/ablation.json`.

| Model | Features | f1_seen | recall_future | dyre recall | matsnu recall |
|---|---|---|---|---|---|
| logreg | 8 basic (M1) | 0.7635 | 0.8003 | 1.0 | 0.0587 |
| logreg | all 15 | 0.8291 | 0.8194 | 1.0 | 0.0123 |
| gb | 8 basic | 0.7861 | 0.6771 | 0.2753 | 0.0207 |
| gb | all 15 (M1, production) | 0.8891 | 0.7885 | 0.646 | 0.0494 |

- **Reproduction.** The two M1 cells reproduce M1 exactly (every metric and every per-family
  recall). A rerun on the same machine writes a byte-identical report.
- **The M1 gap is about half features, half model.** Of the 0.1256 `f1_seen` gap between M1's
  logreg and gb:
  - giving logreg the dictionary and n-gram features closes 0.0656 (0.7635 → 0.8291);
  - with the same 15 features, gb still leads logreg by 0.06 (0.8891 against 0.8291).
  - So "gb is better on seen families" still holds, but by less than the M1 table suggests.
- **Unseen families: on 15 features logreg has the higher recall** (`recall_future` 0.8194 against gb's 0.7885),
  but lower precision on `test_seen` (0.8385 against 0.8929), so it also flags more benign domains.
- **The extra features help gb much more than logreg** (+0.103 `f1_seen` against +0.0656). gb
  on the 8 basic features is the weakest cell on unseen families (`recall_future` 0.6771).
- **dyre:** logreg catches every dyre label with either feature set. gb catches fewer with the 8
  basic features (0.2753) than with all 15 (0.646). See
  [dyre and the tree models](#dyre-and-the-tree-models).

gb stays the production model: it is best on `f1_seen`, the metric declared in advance, also
when logreg gets the same features.

### Per-family recall (gb)

Full tables for all three models: `reports/per_family_recall.json`. For DGA families the value is
recall; for `benign` it is the share predicted benign.

**test_seen** (families seen in training):

- Perfect (1.0): `fobber_v1`, `qadars`, `ramdo`, `ranbyus_v1`, `ranbyus_v2`, `zeus-newgoz`.
- Weakest: `suppobox_2` 0.2228, `vawtrak_v3` 0.6054, `pizd` 0.6931, `suppobox_1` 0.697.
- Benign: 0.8934.
- The dictionary and n-gram features help a lot: logreg, which does not get them, scores 0.0 on
  `pushdo`, `vawtrak_v2` and `vawtrak_v3` (gb: 0.913, 0.8125, 0.6054).

**test_future** (held-out families):

- Caught almost fully: `banjori` 1.0, `murofet_v1` 1.0, `murofet_v3` 1.0, `murofet_v2` 0.9987,
  `sisron` 0.9987, `padcrypt` 0.998, `tinba` 0.9927, `qakbot` 0.9917, `necurs` 0.981.
- Missed: `matsnu` 0.0494, `nymaim` 0.2107, `simda` 0.4315, `dyre` 0.646, `shiotob` 0.7403.

## Explainability and the drift story

`POST /explain` returns the prediction plus the features that pushed it toward `dga` (positive
contribution) or `benign` (negative). For gb the contributions are SHAP values in log-odds, and
`base_value` plus all 15 contributions equals the model's log-odds (checked in the tests).

### matsnu: "hyphen + English words = benign"

The real response for `wedding-muscle.com`, a matsnu domain from the held-out set:

```json
{
  "domain": "wedding-muscle.com",
  "label": "benign",
  "probability": 0.0036,
  "model_version": "20261105_062411",
  "in_scope": true,
  "warning": null,
  "output_units": "log_odds",
  "base_value": -0.03691765562588492,
  "top_contributing_features": [
    {"feature": "hyphen_count", "value": 1.0, "contribution": -3.42574984061764},
    {"feature": "bigram_unseen_ratio", "value": 0.15384615384615385, "contribution": -1.021768467627285},
    {"feature": "trigram_mean_logfreq", "value": -3.9781896608308833, "contribution": -0.5972774506335715},
    {"feature": "bigram_mean_logfreq", "value": -2.837270528388683, "contribution": -0.39355988586259727},
    {"feature": "dict_coverage", "value": 1.0, "contribution": -0.30221298405663727}
  ]
}
```

`hyphen_count` dominates: one hyphen pushes the log-odds down by 3.43, about three times as much as
the next feature. The rest is the label looking like English: common n-grams and full dictionary
coverage.

The reason is in the training data. **None of the training DGA labels contains a hyphen**, while
hyphenated names are common among real benign sites. So the model learned the shortcut
"hyphen + English words = benign". matsnu builds its domains exactly that way, joining dictionary
words with hyphens, so all three models catch only about 5% of it (`matsnu` recall: gb 0.0494,
rf 0.0457, logreg 0.0587).

This is the drift Milestone 2 has to handle. Per-family recall on `test_future` is the baseline
that retraining should improve. Round 1 of the retraining loop added labelled matsnu domains:
matsnu recall rose from 0.056 to 0.873, but the model then flagged hyphenated benign sites too,
and the gate rejected it on the false-positive rate (see [Project Focus](#project-focus)).
Retraining alone swaps one hyphen shortcut for another; hyphen-insensitive features are the
planned fix.

### dyre and the tree models

dyre labels are one letter followed by 33 hex characters (e.g.
`a022b679b3dcd50dda1b4c6633662a3512`), always 34 characters long. Both tree models miss about a
third of them (`dyre` recall: gb 0.646, rf 0.641), while logreg catches all of them (1.0).

**Correction (Milestone 2).** The Milestone 1 README explained this as "a pattern trees cannot
extrapolate to", because 34 characters is longer than any DGA label in training. The length fact
is right, but the explanation is not:

- **dyre is mostly inside the range the trees saw.** 17 training labels are 34 characters or
  longer (up to 54), and all of them are benign. Only `digit_letter_transitions` goes beyond the
  training range (5 to 27 for dyre, at most 20 in training).
- **What pushes gb's misses toward benign is not length.** Mean SHAP contributions over gb's
  1,062 dyre misses in `test_future`:
  - `digit_ratio` −1.66 and `unique_char_ratio` −1.23 log-odds;
  - `length` only −0.23.
  - A label drawn from only 16 hex symbols has a low unique-character ratio, which among long
    training labels was mostly benign. So the trees follow the nearest training examples, and
    those are benign.
- **It depends on the model, not on "trees" in general.** In the
  [ablation](#feature-set-ablation-milestone-2), gb on the 8 basic features catches 0.2753 of
  dyre, fewer than gb on all 15 (0.646). The M1 review expected the opposite (about 0.89 on the
  basic features) from a HistGradientBoosting model with default settings. How a tree model scores
  dyre depends on where its splits fall, which changes with the model and its settings.
- **logreg's score is a weighted sum.** dyre's many digits and letter/digit switches push it
  steadily toward dga, and logreg flags every dyre label with either feature set.

dyre is in the third phase-in batch. Retraining on labelled dyre domains gives the trees DGA
examples in this region.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Classify one domain: `label` (`benign` or `dga`) and P(dga) |
| `POST` | `/explain` | The same prediction plus the top 5 SHAP feature contributions |
| `GET` | `/health` | Model and explainer status: 200 `healthy` or `degraded`, 503 `unhealthy` (used by health checks) |
| `GET` | `/stats` | Request counts, latency and prediction statistics since the service started |
| `GET` | `/docs` | Interactive OpenAPI docs with request and response examples |
| `GET` | `/` | Redirects (307) to `/docs` |

Start the server, then open **http://127.0.0.1:8000/docs** for the interactive OpenAPI docs (with
request and response examples):

```powershell
venv\Scripts\python -m uvicorn src.app:app
```

The model and the explainer are loaded once at startup from `models/production/latest.json` (the
`MODEL_POINTER` environment variable overrides that path; see
[Production model and requirements](#production-model-and-requirements)). If loading fails,
including a failed SHA-256 check, the service still starts: `/health` reports it, and `/predict`
and `/explain` return 503.

The raw input may be at most 1,024 characters, checked before anything else (a domain is at most
253; the rest leaves room for a pasted URL). It is then normalized (trimmed, lowercased; scheme,
path, port and trailing dot removed, so `HTTP://Example.COM/path` becomes `example.com`) and must
be a valid DNS name: ASCII only, at
least one dot, labels of 1-63 letters/digits/hyphens with no leading or trailing hyphen, at most
253 characters, and not a bare public suffix like `co.uk`. It must also end in a known public
suffix: IP addresses (`8.8.8.8`) and internal names (`host.local`, `abc.internal`) are rejected,
because the model never saw them in training and would score the wrong label (for
`pidfksipycmyg.local` it would look at `local`). Anything else gets a 422 with a message.

### Scope warnings

Some valid domains are outside what the model can judge. They still get a prediction, but the
response says `"in_scope": false` and gives a `warning`; for every other domain `in_scope` is
`true` and `warning` is `null`. Both fields are in `/predict` and `/explain`, and `/stats` counts
out-of-scope predictions in `predictions.out_of_scope`.

- **Private suffixes** (C1), e.g. `x.blogspot.com`, `x.duckdns.org`, `x.github.io`: the features
  use the second-level label under the *public* suffix list, which here is the provider's name
  (`blogspot`), so the part a DGA would generate (`x`) is never seen. Detected with `tldextract`
  including the list's private section. The provider's own domain (`blogspot.com`) is in scope.
- **Punycode** (C2): any label starting with `xn--` (`xn--bcher-kva.de`, `example.xn--p1ai`).
  These encode internationalized names, which the ASCII training data does not cover.

```json
{"domain": "abcxyzqwrtplk.blogspot.com", "label": "benign", "probability": 0.0451, "model_version": "20261105_062411", "in_scope": false, "warning": "'abcxyzqwrtplk.blogspot.com' is under the private suffix 'blogspot.com' (a hosting or dynamic-DNS provider); the model only sees the label 'blogspot', so the prediction says nothing about 'abcxyzqwrtplk'."}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"domain": "moiqidfvjjhjmndqp.in"}'
```

```json
{"domain": "moiqidfvjjhjmndqp.in", "label": "dga", "probability": 0.996, "model_version": "20261105_062411", "in_scope": true, "warning": null}
```

```json
{"domain": "teco.com.tw", "label": "benign", "probability": 0.0021, "model_version": "20261105_062411", "in_scope": true, "warning": null}
```

Both example domains are from `test_seen`, so the model never trained on them:
`moiqidfvjjhjmndqp.in` is a `ranbyus_v2` DGA domain and `teco.com.tw` is a Tranco domain.
`probability` is P(dga), rounded to 4 decimals; `label` is `dga` when it is at least 0.5.
`in_scope` and `warning` are explained under [Scope warnings](#scope-warnings).

Invalid input, HTTP 422. For `-bad.com`:

```json
{"detail": [{"type": "value_error", "loc": ["body", "domain"], "msg": "Value error, label '-bad' must not start or end with a hyphen", "input": "-bad.com", "ctx": {"error": {}}}]}
```

For `8.8.8.8`:

```json
{"detail": [{"type": "value_error", "loc": ["body", "domain"], "msg": "Value error, '8.8.8.8' has no known public suffix; IP addresses and internal names are not supported", "input": "8.8.8.8", "ctx": {"error": {}}}]}
```

### `POST /explain`

Same request and validation as `/predict`. The response adds `base_value`, the 5 features with
the largest absolute contribution, and `output_units`: the units of `base_value` and the
contributions, `log_odds` for gb and logreg or `probability` (P(dga)) for rf (see the
`wedding-muscle.com` example above):

```bash
curl -X POST http://127.0.0.1:8000/explain -H "Content-Type: application/json" -d '{"domain": "wedding-muscle.com"}'
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "healthy", "model_loaded": true, "explainer_loaded": true, "git_sha": "67fc13b", "model_version": "20261105_062411"}
```

`git_sha` is the commit the running image was built from (Docker build arg `GIT_SHA`, see
[Docker](#docker)), so a deployment can be identified; it is `"unknown"` when not set, e.g. when
running from a venv.

| `status` | HTTP | When |
|---|---|---|
| `healthy` | 200 | model and explainer loaded |
| `degraded` | 200 | model loaded but the explainer failed: `/predict` works, `/explain` returns 503 |
| `unhealthy` | 503 | no model (`{"status": "unhealthy", "model_loaded": false, "explainer_loaded": false, "git_sha": "..."}`) |

`degraded` stays 200 so that Docker and Render health checks do not restart a service that still
answers `/predict`; the body and the server log say what failed.

### `GET /stats`

Monitoring counters since the service started. It always returns 200, also without a model
(`model_version` is then `null`).

```bash
curl http://127.0.0.1:8000/stats
```

```json
{
  "model_version": "20261105_062411",
  "started_at": "2026-11-25T09:00:00.000+00:00",
  "uptime_seconds": 123.456,
  "requests": {
    "total": 6,
    "by_status_class": {"2xx": 5, "4xx": 1},
    "avg_latency_ms": 7.512,
    "by_endpoint": {
      "GET /health": {"count": 2, "by_status_class": {"2xx": 2}, "avg_latency_ms": 0.811},
      "POST /predict": {"count": 4, "by_status_class": {"2xx": 3, "4xx": 1}, "avg_latency_ms": 10.863}
    }
  },
  "predictions": {"count": 3, "by_label": {"benign": 2, "dga": 1}, "mean_probability_dga": 0.3349, "out_of_scope": 1}
}
```

- `requests` counts every request, per status class (`2xx`, `4xx`, ...) and per endpoint, keyed by
  method and route. Requests that match no route (e.g. scans for `/wp-admin`) are grouped under
  `unmatched`. Latency is measured in the service, from request in to response out.
- `predictions` covers successful `POST /predict` calls only (`/explain` is not counted):
  the count per label, the mean P(dga), and how many were out of scope (`in_scope: false`).
- **Counters are in memory and reset whenever the service restarts.** On Render's free tier an
  idle service is put to sleep and restarted on the next request, so `started_at` shows how long
  the counters cover. The `/stats` call itself is counted after its response is built.
- **Health checks inflate `requests.total`.** Docker's `HEALTHCHECK` and Render's health checks call
  `/health` regularly, so the totals rise with no user traffic. For demos, read `POST /predict`
  under `by_endpoint` and the `predictions` block.

Every request also writes one JSON line to stdout (logger `dgalert.access`), next to uvicorn's own
access line. Request bodies are never logged, so the domain does not appear; `/predict` adds only
the label and probability:

```json
{"timestamp": "2026-11-25T09:00:01.234+00:00", "method": "POST", "path": "/predict", "status": 200, "latency_ms": 9.81, "label": "dga", "probability": 0.996}
```

## Docker

Build the API image and run it (PowerShell or bash, from the repo root):

```bash
docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t dgalert:local .
docker run --rm -p 8000:8000 --name dgalert dgalert:local
```

On Linux/macOS, `make docker-build` and `make docker-run` do the same. The API is then at
http://127.0.0.1:8000 (docs at `/docs`), and `docker ps` shows the container as `(healthy)` about
10 seconds after it starts. Stop it with Ctrl+C or `docker stop dgalert`.

- **Image**: multi-stage build on `python:3.13.18-slim-trixie` (Python 3.13.18, Debian 13), pinned
  to the exact patch release. The builder installs only `requirements-api.txt` into a venv (no
  mlflow, pytest or ruff); the runtime stage copies that venv, `src/`, `data/reference/` (the word
  list the features load) and `models/production/`. About 705 MB unpacked (161 MB compressed).
- **Runs as a non-root user** (`dgalert`, uid 10001) that cannot modify the code, the model or the
  installed packages. `NUMBA_CACHE_DIR=/tmp/numba_cache` gives shap's numba a writable cache folder.
- **Port**: uvicorn listens on `0.0.0.0:${PORT:-8000}`, so a platform that sets `PORT` (Render)
  works unchanged: `docker run --rm -e PORT=9000 -p 9000:9000 dgalert:local`.
- **Health check**: Docker calls `/health` with Python's `urllib` (slim images have no curl) every
  2 seconds while starting and every 10 seconds after. `healthy` and `degraded` (200) pass;
  `unhealthy` (503) fails. These calls show up in `/stats` and in the log.
- **Logs**: one JSON line per request on stdout (`docker logs dgalert`); uvicorn's own access
  line is switched off (`--no-access-log`).
- **`GIT_SHA`**: the build arg is stored in the image and returned by `/health`, which identifies
  the deployed commit. Without it the value is `unknown`.

## Deployment

The API runs on Render (free instance) at **https://dgalert.onrender.com** (docs at
https://dgalert.onrender.com/docs). The Render web service runs the prebuilt image
`ghcr.io/piranchall/dgalert` from GHCR (no build on Render) with health-check path `/health`.

Every merge to `main` deploys itself, with no manual step:

1. CI's `lint` and `test` pass, and `build` builds the image, smoke-tests it and pushes it to GHCR
   as `sha-<short sha>` and `main` (see [CI](#ci)).
2. The `deploy` job runs the reusable workflow `.github/workflows/deploy.yml` with that `sha-`
   tag. It checks that the image exists, then calls the service's deploy hook (GitHub secret
   `RENDER_DEPLOY_HOOK`) with `imgURL=ghcr.io/piranchall/dgalert:sha-<short sha>`, so Render runs
   exactly the image that was tested, not whatever `:main` points to at that moment.
3. It then polls https://dgalert.onrender.com/health every 10 seconds until `git_sha` is the
   merged commit, and fails after about 10 minutes if it never is.

Rollbacks and roll-forwards use the same workflow by hand, with no Render login
(`gh workflow run deploy.yml -f tag=sha-<short sha>`): see [RUNBOOK.md](RUNBOOK.md), which also
covers model rollbacks, running a production image locally and what to check when the health
check fails.

Version tags (`v*`) publish an image but do not deploy. **Cold starts:** a free instance stops
after about 15 minutes without traffic, and the next request waits (up to about a minute) while
it starts again. `/stats` counts from that start, so it resets after every spin-down and every
deploy.

Calling the live service (bash, macOS/Linux, Git Bash):

```bash
curl https://dgalert.onrender.com/health
curl -X POST https://dgalert.onrender.com/predict -H "Content-Type: application/json" -d '{"domain": "moiqidfvjjhjmndqp.in"}'
curl -X POST https://dgalert.onrender.com/explain -H "Content-Type: application/json" -d '{"domain": "wedding-muscle.com"}'
curl https://dgalert.onrender.com/stats
```

In PowerShell, write `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`), and in Windows
PowerShell 5.1 escape the quotes inside the JSON body:

```powershell
curl.exe https://dgalert.onrender.com/health
curl.exe -X POST https://dgalert.onrender.com/predict -H "Content-Type: application/json" -d '{\"domain\": \"moiqidfvjjhjmndqp.in\"}'
curl.exe -X POST https://dgalert.onrender.com/explain -H "Content-Type: application/json" -d '{\"domain\": \"wedding-muscle.com\"}'
curl.exe https://dgalert.onrender.com/stats
```

(PowerShell 7.3 and later pass quotes through unchanged: there, use the bash form of the body,
`-d '{"domain": "moiqidfvjjhjmndqp.in"}'`.)

## Limitations

- **Purely lexical features.** The model sees only the characters of the second-level label: no
  DNS responses, WHOIS age or traffic. Word-based DGAs can look exactly like a real two-word business
  name. `suppobox` is a training family and still partly missed in `test_seen` (gb recall:
  `suppobox_1` 0.697, `suppobox_2` 0.2228, `suppobox_3` 0.9777). Held-out word-based families do
  worse (`matsnu` 0.0494, `nymaim` 0.2107).
- **Dynamic-DNS families are out of scope.** `corebot`, `symmi`, `kraken_v1` and `kraken_v2` are
  excluded because their generated string sits in a subdomain the features never look at.
- **Punycode was excluded from training.** The 1447 `xn--` domains in Tranco were dropped, so the
  model has never seen internationalized domains. The API accepts `xn--` names, but their
  predictions are unreliable.
- **The benign set is skewed toward lower ranks.** Sampling evenly across rank bands means 9 of
  the 10 bands, and so 90% of benign rows, come from below the top 100,000. Less popular domains
  look more random, so more of them are flagged as DGA. This lowers precision, but it is closer to
  what a resolver really sees than a top-N list would be.
- **One hyphen hides a DGA domain.** `scripts/hyphen_evasion.py` takes 2000 `test_seen` DGA domains
  (seed 42) that gb labels dga and inserts one hyphen in the middle of each second-level label
  (`moiqidfvjjhjmndqp.in` -> `moiqidfv-jjhjmndqp.in`). The share still labelled dga is **0.0000**:
  every one of them flips to benign. Our demo domain drops from P(dga) 0.996 to 0.0067, and
  `/explain` puts `hyphen_count` at -7.50 log-odds. The cause is the training data: **0 of the
  76771 training DGA rows** have a hyphen in their second-level label, while hyphens do occur in
  benign names, so the model learned "hyphen = benign". This is also why matsnu is missed (see
  [the drift story](#matsnu-hyphen--english-words--benign)). Milestone 1 expected retraining on
  hyphenated families such as matsnu to fix this. Round 1 showed it does not on its own: the
  matsnu model flagged hyphenated benign sites instead and was rejected (see
  [Project Focus](#project-focus)). The planned fix is hyphen-insensitive features. Run the check
  with `python -m scripts.hyphen_evasion` (needs `data/processed/` and a trained model).
- **False alarms dominate when DGA traffic is rare.** gb's false-positive rate on `test_seen` is
  **0.1066**, derived as 1 minus gb's `benign` recall on `test_seen` (0.8934) in
  `reports/per_family_recall.json`. `test_seen` is about half DGA, but real DNS traffic is mostly
  benign. Take 10,000 domains of which 1% (100) are DGA. With `recall_seen` 0.8853, gb finds about
  89 of the 100, but it also flags about 1,055 of the 9,900 benign domains (9,900 x 0.1066). Only
  about 7.7% of its alerts would be real (precision 88.53 / (88.53 + 1055.34) = 0.0774).
  Milestone 2 added a separate validation split, used to choose the retraining model's parameters
  ([Retraining data](#retraining-data-milestone-2)). Threshold-independent metrics (ROC-AUC) and a
  threshold chosen on validation for a target false-positive rate are planned for the final
  presentation.
- **Some "benign" labels are probably wrong (Tranco label noise).** Tranco ranks popular
  domains; it does not check that they are harmless. Because the benign rows are sampled across
  the whole top-1M list, they include long-tail names that look machine-generated, and some may
  be malicious or parked. An example is `zpsczg.cn`, one of the benign domains newly flagged in
  round 1. Every such domain counts as a false positive when the model flags it, so part of the
  measured false-positive rate (0.1066) may be label noise rather than model error. We have not
  measured how large that part is. Filtering suspicious Tranco tail domains is planned for the
  final presentation.

MLflow saves artifact locations as absolute paths, so the committed M1 store is only read through
a working copy whose paths are rewritten for this machine (`src/m1_snapshot.py`,
`src/mlflow_store.py`); see [MLflow stores and reports](#mlflow-stores-and-reports).

## Changes since the proposal

Quotes are from our Milestone 0 proposal.

1. **Python 3.13, not 3.11.** The proposal's stack table says "Python 3.11" and
   "Docker (python:3.11-slim base)".
   - **Now:** Python 3.13 locally and in CI, and the image is based on
     `python:3.13.18-slim-trixie`.
   - **Why (our decision):** the pinned `numpy` 2.5.4 and `shap` 0.54.1 require Python 3.12 or
     later.
2. **A committed sample instead of the full UMUDGA dataset.** The proposal describes UMUDGA as
   "~30M labeled DGA domains across 50 malware families", and `prepare_data.py` as the step that
   "downloads UMUDGA + Tranco".
   - **Now:** the pipeline uses a committed sample of about 234k domains: 3,000 per family for
     the 46 usable families (138,000 DGA) plus 96,000 Tranco domains. `prepare_data` reads
     `data/sample/` and downloads nothing.
   - **Effect:** every run (local, CI, the retraining workflow) gets the same data without a
     download.
   - The sample is built once by `scripts/make_sample.py` from the raw files
     ([Data](#data)).
3. **chrmor/DGA_domains_dataset is not used.** The proposal planned it "for fast local iteration
   and as a sanity-check source". The committed UMUDGA sample is the only DGA source, and Tranco
   the only benign source.
4. **MLflow uses SQLite stores.** The proposal planned a "local file-backed store, committed to
   repo via mlruns/".
   - **Now:** the M1 runs are a committed, read-only SQLite snapshot (`mlflow.db` + `mlruns/`).
     New runs go to a gitignored local store, and the retraining runs to the workflow artifact.
   - **Why:** MLflow 3 recommends SQLite for local tracking, and a store that every training run
     rewrites does not belong in git (review finding B2).
5. **Models are `.joblib`, not `.pkl`.** The proposal names "models/model_<timestamp>.pkl". The
   production pointer also records the file's SHA-256, and the loader checks it.
6. **`/stats` instead of `/metrics`.** The proposal's `/metrics` was to report "request count,
   average latency, and the rolling distribution of predictions".
   - **Now:** `GET /stats` returns request counts, latency and the prediction counts per label
     since the service started, cumulative rather than rolling, as JSON.
7. **`/explain` is a separate call.** The proposal said it "accompanies every /predict call".
   - **Now:** `/explain` is its own endpoint, returning the same prediction plus the SHAP
     contributions; `/predict` returns the prediction only.
8. **No fresh Tranco sample per retraining round.** The proposal's workflow "pulls a refreshed
   Tranco benign sample" each round.
   - **Now:** every round uses the same Tranco list (`746WX`).
   - **Why:** the gate compares champion and challenger on a frozen evaluation set, and a round
     must be reproducible. With a new benign sample each round:
     - the training data would change in two ways at once, and a change in the false-positive
       rate could not be traced to the new families;
     - new benign rows could overlap the frozen evaluation set.
9. **The promotion gate was redesigned.** The proposal promotes "only if F1 on the frozen test set
   strictly improves and the sanity-test suite passes".
   - **Now:** six criteria ([Promotion gate](#promotion-gate-milestone-2)): recall on the new
     families, F1 on the seen families, false-positive rate, recall on earlier phased-in families,
     the sanity domains and the frozen hash. Both models are scored on the same rows, instead of
     comparing stored numbers.
   - **Why: round 1 shows the old rule would have promoted a worse model.**
     - Overall F1 on the frozen set rose from 0.8870 to 0.8952, and the sanity domains passed
       6/6.
     - So the proposal's rule would have promoted the matsnu model.
     - But its false-positive rate rose from 0.1066 to 0.1235 (396 more benign domains flagged).
       On mostly-benign real traffic that means many more false alarms.
     - The new gate rejected it on exactly that criterion (Actions run 45668715690; see
       [Project Focus](#project-focus)).
10. **Retraining is manual, with a weekly dry run, and promotes through a reviewed PR.** The
    proposal's workflow "runs on a schedule and on manual trigger", and "Promotion rebuilds and
    pushes a new Docker image tag; Render auto-deploys on the new image".
    - **Now:** real rounds start only from a manual run on `main`, and the weekly scheduled run
      is always a dry run.
    - **A promotion opens a pull request** for the other member to review. Merging it builds,
      tests and deploys the new image through the normal CI pipeline.
    - **Why:** a model change goes through the same review and checks as a code change, and it
      can be reverted like one ([RUNBOOK.md](RUNBOOK.md#roll-back-a-model)).
11. **Retraining uses HistGradientBoosting.** The proposal's classifier is "a scikit-learn
    gradient-boosted tree model".
    - **Now:** the retraining loop fits `HistGradientBoostingClassifier`, also gradient-boosted
      trees, with fixed parameters chosen once on the validation split.
    - **Why:** it fits in about 10 seconds, against about 4.4 minutes for the M1
      `GradientBoostingClassifier` (review finding B6).
12. **CI does more than "runs on every push to main".**
    - **Triggers:** it also runs on every pull request and on `v*` tags.
    - **Image:** the image is published to GHCR as `sha-<short sha>`.
    - **Deploy:** a `deploy` job ships that exact image to Render ([CI](#ci),
      [Deployment](#deployment)).
13. **Evidently (listed as optional) is not used.** Drift is measured through per-family recall
    on the frozen evaluation set instead. Drift signals in `/stats` are planned for the final
    presentation.
14. **The UMUDGA citation.** The proposal names "an accompanying paper in Computers & Security".
    The dataset itself is described in *Data in Brief* 30 (2020), 105400, which is the
    reference we use everywhere ([Citations](#citations)).
15. **Reviews.** The proposal's mitigation says "Both members review every PR". What the records
    show is in [CONTRIBUTION.md](CONTRIBUTION.md).

## Tests

```powershell
venv\Scripts\python -m pytest -q
venv\Scripts\ruff check src tests scripts
venv\Scripts\ruff format --check src tests scripts
```

### CI

GitHub Actions (`.github/workflows/ci.yml`, Ubuntu, Python 3.13) runs on every pull request, on
pushes to `main` and on `v*` tags. `lint`, `test` and `build` are the required status checks;
`deploy` runs only on pushes to `main`:

| Job | What it does |
|---|---|
| `lint` | `ruff check` and `ruff format --check` (installs only the ruff version pinned in `requirements.txt`) |
| `test` | `pytest -q` with `requirements.txt`, including the sanity tests on the production model |
| `build` | after `lint` and `test` pass: builds the Docker image with `GIT_SHA`, waits until Docker reports it healthy, checks that `/health` shows that commit and that `/predict` labels `moiqidfvjjhjmndqp.in` as `dga`, then stops it |
| `deploy` | pushes to `main` only, after `build`: runs `deploy.yml` with the image's `sha-` tag, which deploys it to Render and waits until the live `/health` reports the commit (see [Deployment](#deployment)) |

On pushes to `main` and `v*` tags only, `build` then pushes **the image it just tested** to the
GitHub Container Registry as `ghcr.io/piranchall/dgalert:sha-<short sha>`, plus `:main` or the
version tag (e.g. `:v2.0-milestone2`). Pull requests never push. To run a published image:

```bash
docker pull ghcr.io/piranchall/dgalert:main
docker run --rm -p 8000:8000 ghcr.io/piranchall/dgalert:main
```

| File | Covers |
|---|---|
| `tests/test_config.py` | paths inside the repo, local MLflow store and its env-var overrides, working folders gitignored, seed, family lists disjoint |
| `tests/test_features.py` | normalization, second-level label extraction, every feature, dictionary split |
| `tests/test_prepare_data.py` | validation drops, grouped split with no shared labels, held-out families only in test_future, identical output on a rerun |
| `tests/test_gate.py` | promote; reject on each of the six criteria alone; several failures all listed; inclusive boundaries; earlier-family criterion passes with no earlier families and catches a collapse that seen F1 misses; stored metrics ignored; committed hash used by default; configuration errors raise; deterministic JSON; end to end with two small real models on `data/sample` |
| `tests/test_prepare_retrain.py` | hash rule and content-hash format, parts cover the inputs, no label shared between any two parts, about 10% / 20% per family, output independent of row order, membership stable when rows or families are added, one label stays on one side, phase-in batches; on the committed sample: M1 split counts unchanged, frozen eval report equal to the committed one, same hash on a second run |
| `tests/test_train.py` | training on a tiny fixture, MLflow runs, `latest.json` promotion (with `sha256`) and `--force`, report generation, training refuses the committed store and never writes `reports/`, `production_model.json` lookup across stores (deterministic, no paths or times) |
| `tests/test_m1_snapshot.py` | the committed `mlflow.db` hash and `mlruns/` unchanged after the viewer command, the copy relocated and refreshed, a changed snapshot refused, `reports/` rebuilt from the snapshot equal to the committed files (parsed JSON); `reports/production_model.json` matches the production pointer (rebuilt from the snapshot while production is the M1 model) |
| `tests/test_retrain.py` | round planning (next batch, partly phased-in batch refused), training rows, fixed HGB parameters match the search; one round on a small setup from `data/sample` with a temporary production folder and MLflow store: promotion replaces the model (old file deleted, loader and explainer accept the new one, `round` and `phased_in_families` updated, production report from the run), rejection and dry run leave production byte-identical, a second round with earlier families, nothing left to phase in, real gate reasons and MLflow run, frozen-hash mismatch rejects; workflow outputs and Markdown report |
| `tests/test_ablation.py` | the 2 x 2 grid; each pipeline uses exactly its feature set (scrambling the other columns changes nothing), logreg on the basic features is M1's pipeline, gb parameters from M1's best grid point; on small splits from `data/sample`: every cell with overall and per-family numbers, effects computed from the cells, a byte-identical report on a rerun with no run ids, times or paths, one MLflow run per cell, every M1 mismatch listed |
| `tests/test_retrain_workflow.py` | `retrain.yml`: manual dispatch defaults to a dry run, the weekly schedule is always dry, real rounds only on `main`, read-only permissions, step order, the committed frozen report not overwritten, artifact always uploaded, open round PR not duplicated, PR only on a real promotion with `RETRAIN_PR_TOKEN` (title, contents, bot commit, reviewer is the other member), no expressions inside shell scripts |
| `tests/test_model_loader.py` | loading via `latest.json`, the `MODEL_POINTER` override, missing or malformed pointers, SHA-256 check (tampered file refused, missing hash warns), scikit-learn version warning, the committed production model (fields, hash, loading from another working directory) |
| `tests/test_app.py` | `/health` and `/predict`: success, 503 without a model or with a tampered model, every 422 validation rule |
| `tests/test_monitoring.py` | `/stats` counts per endpoint and status class, latency, label counts matching `/predict`, `unmatched` paths, reset on restart, working without a model; thread-safe counters; one JSON log line per request with no request body |
| `tests/test_hardening.py` | scope warnings for private suffixes and punycode (C1, C2) and their `/stats` count, `/health` `degraded` when the explainer fails (C3), the 1,024-character input limit checked before normalization (C4), `output_units` per model type (C5), `GET /` redirect (C6) |
| `tests/test_ci.py` | the CI workflow: jobs `lint`, `test`, `build` (build after the other two), triggers, pinned ruff, `packages: write` only for `build`, publishing only on pushes to `main` and `v*` tags, lowercase image name, smoke test with a timeout, `deploy` only on pushes to `main`, calling `deploy.yml` with the tested `sha-` tag |
| `tests/test_deploy_workflow.py` | `deploy.yml`: manual and CI triggers with a required tag, tag format (accepted and rejected examples), the tag reaching shell only through `env`, image-exists check before the hook, hook from the secret (never echoed) with `imgURL`, wait for the tag's commit with a timeout, one deploy at a time; README Deployment section; `RUNBOOK.md` commands match the workflow, its required sections, placeholders instead of example tags in rollback and roll-forward, the wait before picking the newest run, and the rehearsal record |
| `tests/test_docker.py` | `GIT_SHA` in `/health` (set, unset, and in the 503 body); static checks of the `Dockerfile` (pinned base image in both stages, only `requirements-api.txt`, non-root user, health check, port, command) and `.dockerignore` |
| `tests/test_explain.py` | `/explain` shape and agreement with `/predict`; SHAP additivity for gb, hgb, rf and the logreg fallback |
| `tests/test_hyphen_evasion.py` | hyphen insertion in `scripts/hyphen_evasion.py` keeps the suffix and splits the label in the middle |

The tests run from a clean clone with no `data/raw/`, `data/processed/` or trained candidate in
`models/`: they train tiny models on `data/sample/` or on synthetic data in a temporary folder.
They read the committed M1 snapshot only through copies in a temporary folder, and the run fails
if the committed `mlflow.db` hash changes.

Tests marked `@pytest.mark.sanity` check the committed production model in `models/production/`
(e.g. `teco.com.tw` is benign, `moiqidfvjjhjmndqp.in` is dga). They always run, in CI too. Run them
alone with `pytest -m sanity`.

## Citations

- **UMUDGA**: M. Zago, M. Gil Pérez, G. Martínez Pérez. "UMUDGA: A dataset for profiling
  algorithmically generated domain names in botnet detection." *Data in Brief* 30 (2020), 105400.
  DOI [10.1016/j.dib.2020.105400](https://doi.org/10.1016/j.dib.2020.105400).
  Dataset: Mendeley Data, DOI [10.17632/y8ph45msv8.1](https://doi.org/10.17632/y8ph45msv8.1) (MIT licence).
  This is the reference used everywhere in the project, slides included. (The proposal named
  *Computers & Security* as the venue. That journal has a companion research article; the dataset itself is
  described in the *Data in Brief* paper.)
- **Tranco**: V. Le Pochat, T. Van Goethem, S. Tajalizadehkhoob, M. Korczyński, W. Joosen.
  "Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation." *NDSS* 2019.
  List `746WX`: https://tranco-list.eu/list/746WX/1000000
- **SCOWL** (Spell Checker Oriented Word Lists), K. Atkinson, version 2020.12.07, size 50,
  American English: http://wordlist.aspell.net/. Licence in `data/reference/SCOWL-LICENSE.txt`.
- **tldextract**, J. Kurkowski: https://github.com/john-kurkowski/tldextract
- **SHAP**: S. M. Lundberg, S.-I. Lee. "A Unified Approach to Interpreting Model Predictions."
  *NeurIPS* 2017. S. M. Lundberg et al. "From local explanations to global understanding with
  explainable AI for trees." *Nature Machine Intelligence* 2 (2020).
- **scikit-learn**: F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *JMLR* 12
  (2011), 2825-2830.
- **MLflow**: M. Zaharia et al. "Accelerating the Machine Learning Lifecycle with MLflow."
  *IEEE Data Engineering Bulletin* 41(4), 2018.
- **FastAPI**, S. Ramírez: https://fastapi.tiangolo.com

## Licence

MIT, see [LICENSE](LICENSE). Team: Piranchal (29050) and Muhammad Hussain (28985).
