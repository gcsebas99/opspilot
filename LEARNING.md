# LEARNING.md — my notes, in my own words

One section per concept. 3–5 lines each: what it is, how OpsPilot implements it (file), one tradeoff.
If I can't write it without looking, I don't understand it yet.

## Day 1
### Environment (bounded, reproducible, inspectable)
Environment in confined and well-defined, it can be rebuild (same initial params => same result), similar to a deterministic generator function and it provides ways to access data to inspect the state.
### Tools & registry (schemas, errors back to the model, truncation)
Tools are well defined and registered to be acccessed and used by the model. Errors are always reported back to the model so it can decide next action, if a tool generates larger result content, it can be truncated to avoid exceding token limit.
### Context assembly & skills (progressive disclosure, prompt caching)
Context should include system prompt and runbook index, runbooks should be loaded lazily to avoid context consumption (progressive disclosure), on each turn conversation and tools result are included to the messages list. Initial prompts can be cached to save context window.
### ReAct loop & exit conditions
Loop reasonate - take actions - observer - repeat. There might be multiple exit conditions: final outcome, token limit exceeded, max attemps exceeded, agent stucked (same tool attept several times), model not following expected behaviour (just chatting, not using tools).
### Retries / error handling
Retries should be limited, loop should end even with non-positive outcome, but preventing unnecessary waste of resources. Error should be logged properly for observability (evaluations). 429 rate-limit and 5xx/529-overloaded are retried. 400/401/404 -> stops immediately, issue in our end. Retries with exponential backoff + jitter

## Day 2
### Tracing & metrics
### LangGraph vs raw loop
### Permissions
### HITL & checkpointing
### Guardrails / prompt injection
### Audit log

## Day 3
### Golden dataset design
### Grader layers (trajectory, outcome, judge, budget)
### pass@k vs pass^k
### Record / replay
### Judge calibration
### Ablation findings

## Day 4
### Outer loops (event, cron, heartbeat)
### Compaction
### Short-term vs long-term memory
### Deploy tradeoffs

## Interview answers I want to nail
-

### from agent review - day 1

  One actual error to fix

  Context assembly: "Initial prompts can be cached to save context window." — this is the one factual mistake. Prompt caching does not reduce the context window or how many tokens get sent — the full
  prefix is still there. What it saves is cost and latency: a cached prefix is billed at a fraction of the price (~10% for cache reads vs full price) and returns faster, because Anthropic's servers
  reuse the already-processed KV-cache state instead of reprocessing those tokens. If an interviewer asks "what does prompt caching save," "context window" is the wrong answer — cost/latency is the
  right one.

  Per-section: what's right, what's thin

  Environment — Correct at the concept level, but missing the mechanism, which is what an interviewer actually probes: bounded = the Sandbox.path() traversal guard (rejects .., absolute paths, and
  resolves symlinks to check they stay inside root); reproducible = one random.Random(seed) threaded through every call site (never the global random module) plus a fixed epoch instead of
  datetime.now(). "Same params → same result" is the what; you'll want the how ready too.

  Tools & registry — Good, but conflates two distinct failure modes that are actually separate [HARNESS:*] concepts in the code: (1) bad args → pydantic validation error → ok=False with a
  model-readable message so the model can self-correct on its next turn, vs (2) a tool that raises → caught at the registry boundary so one buggy tool never crashes the whole loop. Also missing:
  truncation isn't silent — it appends an explicit "...[N more lines truncated, narrow your pattern]" hint, so the model knows it didn't see everything (this is literally one of the spec's named
  interview questions — "how do you tell the model it was truncated?" — worth having a crisp answer). Also unmentioned entirely: tool descriptions are prompt engineering (write them like docs for a
  new engineer) — another named interview question with no answer in your notes yet.

  ReAct loop & exit conditions — All 5 conditions are actually there, nice. Two things worth adding: (1) there's a repeated pattern across two of the five — both stuck-detection and the no-tool-call
  condition give the model one nudge before terminating, rather than stopping cold on the first offense. Naming that as a deliberate, reused design principle is a stronger answer than listing 5
  unrelated rules. (2) Nothing about parallel tool calls (multiple tool_use blocks in one turn, all executed, all tool_results returned in a single message preserving tool_use_id pairing) or the
  permission stub (destructive tools denied unless --allow-destructive, living in the loop rather than the registry) — both are core loop behavior an interviewer could reasonably ask about.

  Retries / error handling — This is the section I'd rewrite. Right now it reads like it's actually describing the loop's exit conditions again (previous section), not retry logic specifically — worth
  separating those two concepts clearly in your head: retrying a single failed API call (1.5, tenacity) is a different layer from the loop deciding to stop (1.6). Missing entirely: what gets retried
  (429 rate-limit and 5xx/529-overloaded) vs what surfaces immediately (400/401/404 — those are bugs in our own request, not transient, so retrying would just waste time); how retries are spaced
  (exponential backoff + jitter, specifically to stop many concurrent runs from retrying in lockstep and re-triggering the same rate limit); and a genuinely good interview anecdote sitting right there
  unused — while building this, the docs said 529 "overloaded" was a subtype of the generic 5xx error class, and it turned out to be a sibling class in the actual installed SDK. Only caught because
  the test ran against the real SDK instead of trusting the docs. "Tell me about a time you found a bug by testing rather than assuming" is an extremely common interview prompt — you already have the
  story.

  If you do one thing before an interview

  Fill in that empty "Interview answers I want to nail" section at the bottom — docs/specs/day1.md has 7 explicit questions (ReAct loop at the API level, exit conditions, tool error handling, why
  truncate, what makes an environment "good," progressive disclosure, what caching does/breaks). Writing a tight 2-3 sentence answer to each of those now, while it's fresh, is worth more than
  polishing the summary notes further.