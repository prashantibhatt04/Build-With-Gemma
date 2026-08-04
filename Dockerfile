# Real deployment image - ROADMAP_TO_PRODUCT.md Phase 5. One image, three
# real entry points (see docker-compose.yml's `command:` overrides):
# scripts/dashboard.py, scripts/scheduler.py, or an ad hoc `python
# scripts/run_demo.py --auto` for a one-off check - all already-tested,
# already-live-verified code, nothing Docker-specific added inside them.
#
# 3.12 to match .github/workflows/tests.yml's own pinned CI version - the
# same interpreter this project's test suite already runs against, not a
# second, undocumented version choice.
FROM python:3.12-slim

# Real bug this fixes: Python block-buffers stdout when it isn't
# attached to a real terminal (always true inside a container), so
# scripts/scheduler.py's own real per-tick progress prints (tick N:
# screening..., logged N entries...) never actually appear in `docker
# compose logs -f scheduler` until a buffer flush happens - which may
# never visibly occur within a live demo's timeframe. The scheduler is
# genuinely running the whole time (confirmed via its real heartbeat
# file, see scripts/scheduler.py) - this only fixes what's VISIBLE to
# someone watching the logs live, not the underlying behavior.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# Real runtime state (TLE cache, RAG embedding cache, JSONL logs when
# DATABASE_URL isn't set) - not baked into the image, so it survives a
# rebuild and isn't shared across containers via the image layer itself.
VOLUME ["/app/data", "/app/logs"]

# Only scripts/dashboard.py actually serves anything - scheduler.py and
# run_demo.py are one-shot/long-running-without-a-port processes,
# selected via docker-compose.yml's per-service `command:` instead of a
# second Dockerfile.
EXPOSE 8501

CMD ["streamlit", "run", "scripts/dashboard.py", "--server.address=0.0.0.0"]
