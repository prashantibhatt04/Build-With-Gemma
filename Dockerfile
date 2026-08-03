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
