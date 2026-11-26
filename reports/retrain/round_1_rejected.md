## Retraining round 1: **REJECT**

- New families: matsnu
- Earlier families: none
- Training rows: 140657 (train_fit 138248 + phase-in pool 2409)
- Champion: `20261105_062411` (gb); challenger: hgb with fixed parameters
- Frozen eval set: 46800 rows, sha256 `78f48f6b9743`
- MLflow run: `781f46c202e54a3d82b90562749c5f45` (experiment `dgalert-m2-kaizen`)

| Criterion | Champion | Challenger | Change | Threshold | Result |
|---|---|---|---|---|---|
| frozen_eval_hash |  |  |  |  | pass |
| new_family_recall | 0.056027 | 0.872666 | +0.8166 | 0.02 | pass |
| seen_f1 | 0.889092 | 0.885033 | -0.0041 | 0.01 | pass |
| false_positive_rate | 0.106613 | 0.123485 | +0.0169 | 0.005 | **FAIL** |
| prior_family_recall |  |  |  | 0.02 | pass |
| sanity_domains | 6/6 | 6/6 |  |  | pass |

Reasons:

- pass: frozen eval set hash matches reports/eval_frozen.json
- pass: recall on the new families changes by +0.8166 (needs at least +0.02)
- pass: F1 on the seen families changes by -0.0041 (may drop by at most 0.01)
- FAIL: false-positive rate changes by +0.0169 (may rise by at most 0.005)
- pass: no families phased in during earlier rounds: passes
- pass: every sanity domain classified correctly

| Phased-in family | Champion recall | Challenger recall |
|---|---|---|
| matsnu | 0.0560 | 0.8727 |
