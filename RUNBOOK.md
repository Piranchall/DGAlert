# DGAlert RUNBOOK

How to deploy, roll back and check the live service. Every procedure here works **without a
Render login**: you need only `gh` logged in as a member with write access to the repo (either of
us), and Docker for the local checks. Commands are for PowerShell on the shared laptop. In bash,
write `curl` for `curl.exe`.

| | |
|---|---|
| Live service | https://dgalert.onrender.com (docs at `/docs`) |
| Images | `ghcr.io/piranchall/dgalert:sha-<short commit>` (public), one per commit on `main` |
| Deploy workflow | `.github/workflows/deploy.yml` ("Deploy to Render"): deploys one `sha-` tag and waits until `/health` reports that commit |
| What is live | `curl.exe -s https://dgalert.onrender.com/health`: `git_sha` is the commit, `model_version` the model |

## How deployment works

1. Every merge to `main` runs CI. `build` tests the image and pushes it as `sha-<short commit>`
   (and `:main`). Then `deploy` calls `deploy.yml` with that `sha-` tag.
2. `deploy.yml` makes four checks or calls, in order:
   1. It checks the tag format (`sha-` plus 7 lowercase hex characters).
   2. It checks that the image exists in GHCR.
   3. It calls the Render deploy hook (secret `RENDER_DEPLOY_HOOK`) with `imgURL=<that image>`.
   4. It polls `/health` every 10 s until `git_sha` matches, and fails after about 10 minutes.
3. The same workflow is run by hand for rollbacks and roll-forwards:
   `gh workflow run deploy.yml -f tag=sha-<short commit>`.

Things to know:

- **Each image contains its model.** `models/production/` is copied into every image, so
  deploying an older `sha-` tag also brings back the model that commit had.
- **`sha-` images exist only from S6 onward** (`sha-50947c7`, the merge of PR #35, is the oldest),
  and only those images report `git_sha` in `/health`. An older commit has no image, so the
  workflow stops at "Check that the image exists in GHCR". Even if such an image existed, its
  `/health` would never report the commit, so the wait step would time out.
- **Deploys run one at a time.** Manual runs and CI share the concurrency group `render-deploy`,
  so a rollback started during a merge deploy waits for it. If several deploys are waiting, GitHub
  keeps only the newest waiting one.
- **A rollback lasts until the next deploy.** The next merge to `main` deploys `main` again. A
  deploy started from the Render dashboard, or a hook call without `imgURL`, deploys the service's
  configured tag `:main`. Both undo a rollback.

## Normal redeploy

**Automatic:** merge a PR into `main`. Confirm it went live:

```powershell
git switch main; git pull --ff-only
git rev-parse --short=7 HEAD                      # the commit that should be live
$run = gh run list --branch main --workflow CI --limit 1 --json databaseId --jq '.[0].databaseId'
gh run watch $run --exit-status                   # lint, test, build, deploy all green
curl.exe -s https://dgalert.onrender.com/health   # git_sha must equal the commit above
```

**Manual** (for example after a Render incident, to redeploy what is on `main`):

```powershell
git switch main; git pull --ff-only
$tag = "sha-" + (git rev-parse --short=7 HEAD)
gh workflow run deploy.yml -f tag=$tag
```

Then watch the run as in [Watching a deploy run](#watching-a-deploy-run).

## Roll back the service

Use this when the live version misbehaves (errors, wrong predictions, `unhealthy`) and the
previous version was fine.

1. **Note what is live now** (you need it to roll forward):

   ```powershell
   curl.exe -s https://dgalert.onrender.com/health
   ```

2. **Pick the target commit.** Only commits on `main`'s first-parent line have images, and only
   from `50947c7` (S6) onward:

   ```powershell
   git log --first-parent --format="%h %ad %s" --date=format:"%Y-%m-%d %H:%M" main
   ```

   Usually the target is the line below the live commit. Optionally, check that its image exists:
   `docker manifest inspect ghcr.io/piranchall/dgalert:sha-<short>`. It prints JSON if the image
   exists and fails with "manifest unknown" if not. You can also open the package page:
   https://github.com/users/Piranchall/packages/container/package/dgalert.

3. **Deploy it**, for example:

   ```powershell
   gh workflow run deploy.yml -f tag=sha-50947c7
   ```

   You can also start it from GitHub: Actions → "Deploy to Render" → "Run workflow" → tag.

4. **Watch the run** ([below](#watching-a-deploy-run)) until it is green. Then confirm:

   ```powershell
   curl.exe -s https://dgalert.onrender.com/health   # git_sha = 50947c7
   ```

5. **Fix `main`.** The rollback only lasts until the next merge. Revert the bad PR (or merge a fix)
   before anyone merges anything else:

   ```powershell
   git switch main; git pull --ff-only
   git switch -c fix/revert-<topic>
   git revert -m 1 <merge commit of the bad PR>
   ```

   Then open a PR as usual. Merging it deploys the fixed `main` automatically.

## Roll forward

To go back to the newer version after a rollback (for example in a rehearsal, or when the problem
turned out to be elsewhere), deploy the newer tag the same way:

```powershell
gh workflow run deploy.yml -f tag=sha-4e822c5
```

Watch the run and confirm `git_sha` in `/health`. Any merge to `main` also rolls forward
automatically to the new `main`.

## Watching a deploy run

```powershell
# The new run takes a few seconds to appear after "gh workflow run".
gh run list --workflow deploy.yml --limit 3
$run = gh run list --workflow deploy.yml --limit 1 --json databaseId --jq '.[0].databaseId'
gh run watch $run --exit-status
gh run view $run --log-failed     # only if it failed: shows the failing step's output
```

A green run means `/health` already reported the requested commit. The run summary records the
image and the live commit.

## Roll back a model

Each image contains its model, so rolling back a model is first an image rollback.

1. **Roll back the image** to the last commit whose model was good ([Roll back the
   service](#roll-back-the-service)). To see which merges changed the model:

   ```powershell
   git log --first-parent --format="%h %ad %s" --date=format:"%Y-%m-%d %H:%M" main -- models/production
   ```

   The good model is in every image from its own merge up to the merge before the bad model.
   Confirm with `model_version` in `/health` after the deploy:

   ```powershell
   curl.exe -s https://dgalert.onrender.com/health   # model_version = the good model
   ```

2. **Revert the retraining PR that promoted the bad model** (retraining PRs exist from S11 onward)
   with `git revert -m 1 <its merge commit>` in a new PR, so that the next merge does not redeploy
   the bad model. The revert restores the previous `models/production/` files. Merging it deploys
   automatically. Check `model_version` in `/health` again.

**Do not revert #30** (S1, "Production model in the repo"). It added `models/production/`, so
reverting it deletes the only production model: the next image would start `unhealthy` (503, no
model) and the deploy would time out.

## Run a production image locally

Any `sha-` tag (or `:main`) runs locally without logging in, because the package is public:

```powershell
docker pull ghcr.io/piranchall/dgalert:sha-4e822c5
docker run --rm -p 8000:8000 --name dgalert-prod ghcr.io/piranchall/dgalert:sha-4e822c5
```

In a second terminal:

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{\"domain\": \"moiqidfvjjhjmndqp.in\"}'
docker logs dgalert-prod
```

Stop it with Ctrl+C or `docker stop dgalert-prod`. `docker ps` shows `(healthy)` about 10 seconds
after the start. This is the quickest way to tell whether a problem is in the image or on Render.

## Free tier: cold starts and `/stats`

- **Cold starts:** the free instance stops after about 15 minutes without traffic. The next
  request waits while it starts again, up to about a minute. A slow first answer is therefore
  normal and not an incident: retry after a minute before doing anything else. The deploy
  workflow's 10-minute wait already allows for this.
- **`/stats` resets** whenever the process restarts: after every deploy, rollback and spin-down.
  It covers only the time since `started_at`, so read it together with `uptime_seconds`. Render's
  health checks and the deploy workflow's polling also count in `requests.total` (as `GET
  /health`).

## When the health check fails

Start with what the live service says:

```powershell
curl.exe -s -w "`nHTTP %{http_code}`n" https://dgalert.onrender.com/health
```

| What you see | Meaning | What to do |
|---|---|---|
| No answer for up to about a minute, then 200 | cold start | nothing |
| 200 `healthy`, but `git_sha` is older than expected | the newest deploy did not go live: Render keeps the previous version running when a new one fails its health check | check the deploy run (below), then run the new image locally to find the error |
| 200 `degraded` (`explainer_loaded: false`) | model loaded, SHAP explainer failed: `/predict` works, `/explain` returns 503 | roll back if `/explain` is needed, then investigate locally |
| 503 `unhealthy` (`model_loaded: false`) | no model in the image, or it failed to load | roll back now, then investigate locally |
| Still no answer after a few minutes | service down or failing to start | roll back; if the rollback run also times out, the problem is on Render (status page: https://status.render.com), so ask the Render account owner (Piranchal) to check the dashboard |

Then look at the deploy run:

```powershell
gh run list --workflow deploy.yml --limit 3      # manual deploys
gh run list --branch main --workflow CI --limit 3  # automatic deploys (job "deploy")
gh run view <run id> --log-failed
```

| Failing step and message | Cause | Fix |
|---|---|---|
| Check the tag format: "the tag must be sha- followed by 7 lowercase hex characters" | typo, uppercase, or a tag like `main` or `v2.0-milestone2` | use `sha-` + the 7-character commit from `git log` |
| Check that the image exists in GHCR: "not found in GHCR" | the commit is not on `main`, is older than S6, or its CI run failed before the push | pick another commit; check that commit's CI run |
| Trigger the Render deploy hook: "the RENDER_DEPLOY_HOOK secret is not set" | secret missing | the repo admin adds it under Settings → Secrets and variables → Actions |
| Trigger the Render deploy hook: "HTTP 4xx" | the hook URL is wrong or was regenerated | the Render account owner copies the hook URL again into the secret |
| Wait until the live service reports this commit: "did not report git_sha … in time" | Render did not start the new image (failed health check or failed pull), or a very slow start | run that tag locally ([above](#run-a-production-image-locally)) and check `/health` and `docker logs`; if the image is fine locally, rerun the deploy once |

## Rehearsal record

The rollback and roll-forward rehearsals, and the other member following this RUNBOOK unaided,
are recorded here after this RUNBOOK is merged (the deploy workflow must be on `main` before it
can be run by hand).

| # | Who | Action | Tag deployed | Started (PKT) | Live (`git_sha`, time) | Run |
|---|---|---|---|---|---|---|
| | | | | | | |
