# Disk Exhaustion

## Symptoms

- `disk_pct` for a service (often `db`) near 100%.
- Logs containing "no space left on device" or similar.
- Possibly noisy, unrelated WARN spam on *other* services at the same time
  -- don't let that noise pull your investigation away from the actual
  disk metric.

## Likely cause

The affected service's disk is full. This is a resource-exhaustion
incident, not a config change or bad deploy -- `config_history` and
`list_deploys` for the affected service will look unremarkable.

## Diagnostic steps

1. `query_metrics(service, "disk_pct")` -- confirm it's pinned near 100%.
2. `grep_logs(service, "no space left")` (or similar) -- confirm the
   symptom in logs, not just the metric.
3. Check other services' logs/metrics only to confirm they're a red herring
   (normal metrics, just noisy logs) -- don't chase them as the root cause.

## Fix

**There is no safe tool for this.** None of `restart_service`,
`rollback_config`, or `rollback_deploy` frees disk space or fixes a full
disk. Call `escalate` with the disk metric and service as evidence --
don't restart or roll back something unrelated just to feel like you did
something.

## When this isn't it

If `disk_pct` is normal but another service is noisy, that's the red
herring, not this runbook -- keep looking at metrics/logs for what the
alert actually describes.
