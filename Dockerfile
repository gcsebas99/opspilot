# OpsPilot web app -- replay-first public demo (see docs/specs/day4.md).
# Same patch version as .python-version (what tests run on) -- uv refuses a mismatch.
FROM python:3.12.8-slim

# uv pinned to the version that produced uv.lock (CI pins the same one), so
# the image resolves exactly what tests ran against.
COPY --from=ghcr.io/astral-sh/uv:0.9.29 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, in their own layer: a code-only change reuses this
# cached layer instead of reinstalling everything.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Non-root: a compromised process can't modify the app or the system. Only
# tmp/ (per-run sandboxes) is writable -- an ephemeral scratch dir on Render.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/tmp \
 && chown -R app:app /app/tmp
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    OPSPILOT_MODEL_MODE=replay
EXPOSE 10000

# Render sets $PORT (default 10000). --proxy-headers makes request.client the
# real visitor IP (rate limit) behind Render's proxy; trusting any forwarder
# ('*') is safe only because nothing but Render's proxy can reach the container.
CMD ["sh", "-c", "exec uvicorn opspilot.web.app:create_app --factory --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips='*'"]
