# Deploying OpsPilot (Render + MongoDB Atlas, $0)

The public demo runs in **replay mode**: model responses come from recorded cassettes, and
everything else (tools, permissions, approvals, traces, audit) runs for real. The server holds
**no Anthropic key**. How and why it's built this way: [Production](learn/paths/08-production.md).

## Steps

1. **Atlas** — create a free **M0** cluster (AWS `us-west-2`, next to Render's Oregon region);
   *Database Access* → a dedicated user (`opspilot-app`, generated password, read/write on the
   `opspilot` database only); *Network Access* → `0.0.0.0/0`; *Connect → Drivers* → copy the
   `mongodb+srv://…` string.
2. **Render** — *New → Blueprint* → this repo ([`render.yaml`](../render.yaml)). Paste the Atlas
   string when asked for `MONGODB_URI`. Render generates `OPSPILOT_WEBHOOK_SECRET`. Deploys happen
   only after CI passes.
3. **Canary** — in the GitHub repo settings, add the secrets `OPSPILOT_URL` (the Render URL) and
   `OPSPILOT_WEBHOOK_SECRET` (the generated one), then run the *canary* workflow once by hand. It
   also runs weekly.

Run the same image locally:

```bash
docker build -t opspilot . && docker run -p 10000:10000 -e OPSPILOT_STORE=memory -e OPSPILOT_MODEL=claude-haiku-4-5 opspilot
```

## Tradeoffs, written down on purpose

- **Atlas network access is `0.0.0.0/0`.** Render's free tier has no static outbound IPs, so an IP
  allowlist isn't possible. Mitigated by a dedicated least-privilege DB user with a long generated
  password, TLS (always on for Atlas), and nothing sensitive in the database (demo data only).
  With a paid plan: static egress IPs and an allowlist, or a private endpoint.
- **Sandboxes are temp directories, not containers.** Each run gets `tmp/web_runs/<run_id>`, and
  every tool reaches files only through the path guard in `opspilot/env/sandbox.py` — no
  Docker-in-Docker on a free host. Real multi-tenant use would need per-run containers or VMs.
- **The disk is ephemeral** (wiped on restart and spin-down). Conversations survive in the Mongo
  checkpointer; a lost sandbox is rebuilt from `(scenario, seed)` only if the audit log shows no
  destructive action ran yet — otherwise resuming is refused rather than run on the wrong state.
- **Cold starts.** A free service sleeps after 15 idle minutes and takes about a minute to wake;
  Render shows its own loading page meanwhile.
- **One instance.** The per-IP rate limiter lives in process memory; the daily cap lives in Mongo.
  Scaling out would move the rate limiter to a shared store.
- **Proxy headers.** uvicorn runs with `--proxy-headers --forwarded-allow-ips='*'` so the rate
  limiter sees real visitor IPs; trusting any forwarder is only safe because nothing but Render's
  proxy can reach the container.
