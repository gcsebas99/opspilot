# Bad Deploys

## Symptoms

- `error_rate` for a service jumps sharply (not a gradual drift) and stays
  elevated.
- The jump happens close to a deploy timestamp for that same service.

## Likely cause

The most recent deploy introduced a regression. Correlation with deploy
timing is the key signal -- a config change or resource issue usually looks
different (gradual, or tied to a config/metric change instead).

## Diagnostic steps

1. `query_metrics(service, "error_rate")` -- find exactly when the jump
   started.
2. `list_deploys(service)` -- check whether a deploy landed at or just
   before that timestamp.
3. `grep_logs(service, <error pattern>)` -- confirm the new errors reference
   the deployed change (e.g. a specific gateway/timeout/exception type that
   wasn't present before).

## Fix

`rollback_deploy(service)` to redeploy the previous version. This is the
right move once the deploy timestamp and the error_rate jump line up --
don't wait for a root-cause explanation of *why* the new version is bad if
the correlation is already clear.

## When this isn't it

If error_rate rose well before or well after the nearest deploy, look for a
config change or resource exhaustion instead -- don't roll back a deploy
that isn't actually correlated with the symptom.
