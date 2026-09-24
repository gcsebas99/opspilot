# Memory Leaks

## Symptoms

- `mem_mb` for a service climbs roughly linearly over the observed window,
  with no corresponding traffic increase.
- `grep_logs` shows OOMKilled lines, usually clustered near the end of the
  window as the service actually runs out of memory.

## Likely cause

A memory leak in the service itself -- not a config or deploy issue. There
is no metric or config change that "caused" this the way a pool-size edit
or bad deploy would; the growth is intrinsic to the running process.

## Diagnostic steps

1. `query_metrics(service, "mem_mb")` -- confirm a sustained upward trend
   (not a one-time step) over the full window.
2. `grep_logs(service, "OOMKilled")` -- confirm actual restarts/crashes.
3. `list_deploys(service)` -- rule out a recent deploy as the trigger; a
   leak that predates the most recent deploy points to the running process,
   not the code that just shipped.

## Fix

`restart_service(service)` clears the leaked memory and is a legitimate
short-term mitigation. It is **not** a real fix -- the leak will recur.
Recommend escalation (in your report's `recommendation`) so a human
schedules the actual code fix; don't imply the incident is fully resolved.

## When this isn't it

A sudden `mem_mb` jump right after a deploy is a bad deploy, not a leak --
check `list_deploys` before assuming this runbook applies.
