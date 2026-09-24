# DB Connection Pool Issues

## Symptoms

- `latency_p95_ms` rising for a service that talks to the database, often a
  step-change rather than a gradual climb.
- `grep_logs` on the affected service shows lines like "connection pool
  timeout: could not acquire connection".
- `db_pool_in_use` metric pinned at (or very near) its configured max.

## Likely cause

A config change reduced `db_pool_size` for the affected service. Under
normal traffic, a smaller pool queues/times out connections once concurrent
requests exceed the new limit.

## Diagnostic steps

1. `query_metrics(service, "latency_p95_ms")` -- confirm the spike and find
   when it started.
2. `query_metrics(service, "db_pool_in_use")` -- check if it's saturated.
3. `config_history(service)` -- look for a `db_pool_size` decrease whose
   timestamp lines up with the metric spike.

## Fix

`rollback_config(service, <previous_version>)` to restore the prior
`db_pool_size`. Don't guess at a "better" pool size -- roll back to the
version that was known-good before the change.

## When this isn't it

If `db_pool_in_use` is low but latency is still high, the bottleneck is
elsewhere (a slow downstream call, CPU, etc.) -- don't roll back a config
that isn't actually saturated.
