"""Backend-agnostic client for talking to Gemma, local or hosted."""
from __future__ import annotations

import time
from typing import Optional

import requests

from .config import Settings, settings as default_settings

# Circuit breaker for generate() (see GemmaClient._circuit_is_open and
# friends): after this many CONSECUTIVE failed generate() calls against a
# given backend, that backend is treated as "open" - real HTTP attempts
# against it are skipped entirely for CIRCUIT_BREAKER_COOLDOWN_S, rather
# than every single subsequent event paying the full retry+timeout cost
# against a backend that's already known to be down. Counted per
# generate() INVOCATION (after its own same-backend retry is already
# exhausted), not per raw HTTP attempt - 3 separate real-world events all
# finding the backend down in a row, not 3 attempts within one call.
CIRCUIT_BREAKER_THRESHOLD = 3
CIRCUIT_BREAKER_COOLDOWN_S = 60.0


class GemmaClientError(RuntimeError):
    """Raised when the configured Gemma backend cannot be reached or errors out."""


class GemmaClient:
    """Routes generate() calls to either Ollama or a hosted Gemini-style API.

    Callers never need to know which backend is active - they just call
    generate() and get text back, or a GemmaClientError if the backend
    is unreachable or misconfigured.
    """

    def __init__(self, settings: Settings = default_settings):
        self.settings = settings
        # Which backend actually served the most recent successful
        # generate() call - "ollama" or "api". Differs from
        # settings.gemma_backend only when cross-backend fallback kicked in.
        # Callers (see pipeline._describe_model_used) read this for
        # provenance; None until the first successful call.
        self.last_backend_used: Optional[str] = None
        # Circuit breaker state, per backend name ("ollama"/"api") - see
        # module-level constants and _circuit_is_open/_record_success/
        # _record_failure below. Deliberately per-INSTANCE, not shared
        # across GemmaClient objects or processes - matches every other
        # piece of state on this class (last_backend_used included).
        self._consecutive_failures: dict[str, int] = {}
        self._circuit_open_until: dict[str, float] = {}

    def _circuit_is_open(self, backend: str) -> bool:
        open_until = self._circuit_open_until.get(backend)
        return open_until is not None and time.monotonic() < open_until

    def _record_success(self, backend: str) -> None:
        self._consecutive_failures[backend] = 0
        self._circuit_open_until.pop(backend, None)

    def _record_failure(self, backend: str) -> None:
        count = self._consecutive_failures.get(backend, 0) + 1
        self._consecutive_failures[backend] = count
        if count >= CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until[backend] = time.monotonic() + CIRCUIT_BREAKER_COOLDOWN_S

    def _resolve_backend_call(self, backend: str):
        if backend == "ollama":
            return self._generate_ollama
        if backend == "api":
            return self._generate_hosted_api
        raise GemmaClientError(f"Unknown GEMMA_BACKEND '{backend}', expected 'ollama' or 'api'")

    @staticmethod
    def _other_backend(backend: str) -> Optional[str]:
        return {"ollama": "api", "api": "ollama"}.get(backend)

    def _backend_configured(self, backend: str) -> bool:
        if backend == "api":
            return bool(self.settings.gemma_api_key)
        if backend == "ollama":
            return bool(self.settings.ollama_host)
        return False

    def generate(
        self, prompt: str, system: Optional[str] = None, timeout: int = 120,
        format: Optional[dict] = None,
    ) -> str:
        """format, when given, is a JSON Schema object passed through to
        Ollama's own schema-constrained generation (real constrained
        decoding, not a post-hoc parsing hint) - see _generate_ollama.
        Ollama-only: the hosted API path ignores it silently rather than
        raising, so a call that starts on Ollama and falls over to the
        hosted backend (see below) degrades to unstructured text instead
        of failing outright - callers requesting format should still
        handle a response that isn't valid JSON."""
        primary_backend = self.settings.gemma_backend
        # Unknown-backend config errors fail fast - not transient, so no
        # retry and no cross-backend fallback attempt for those.
        backend_call = self._resolve_backend_call(primary_backend)

        last_error: GemmaClientError
        if self._circuit_is_open(primary_backend):
            # Skip the real HTTP retry entirely - this backend has already
            # failed CIRCUIT_BREAKER_THRESHOLD times in a row recently, so
            # a real attempt is very likely to just pay the full
            # retry+timeout cost again for the same known-down backend.
            last_error = GemmaClientError(
                f"Circuit breaker open for '{primary_backend}' after "
                f"{CIRCUIT_BREAKER_THRESHOLD} consecutive failures - skipping retry "
                f"for {CIRCUIT_BREAKER_COOLDOWN_S:.0f}s cooldown"
            )
        else:
            for _ in range(2):  # original attempt + one same-backend retry
                try:
                    text = backend_call(prompt, system, timeout, format)
                    self.last_backend_used = primary_backend
                    self._record_success(primary_backend)
                    return text
                except GemmaClientError as exc:
                    last_error = exc
            self._record_failure(primary_backend)

        # Primary backend exhausted its retry (or its circuit was already
        # open). Before falling through to the caller's deterministic
        # fallback, try the other backend once, if it's actually
        # configured (e.g. don't attempt "api" with no key) and its own
        # circuit isn't open too.
        fallback_backend = self._other_backend(primary_backend)
        if fallback_backend is not None and self._backend_configured(fallback_backend):
            if self._circuit_is_open(fallback_backend):
                raise GemmaClientError(
                    f"Both backends unreachable - {primary_backend}: {last_error}; "
                    f"{fallback_backend}: circuit breaker open after "
                    f"{CIRCUIT_BREAKER_THRESHOLD} consecutive failures"
                )
            fallback_call = self._resolve_backend_call(fallback_backend)
            try:
                text = fallback_call(prompt, system, timeout, format)
                self.last_backend_used = fallback_backend
                self._record_success(fallback_backend)
                return text
            except GemmaClientError as fallback_error:
                self._record_failure(fallback_backend)
                # Both failed - raise a combined error naming both reasons.
                # Raising just last_error (the primary's) here would silently
                # discard the fallback's actual failure reason, which is
                # usually the more useful one to see: the primary is often
                # deliberately/expectedly down (e.g. this project's own
                # failover demo step), so it's the fallback's error that
                # actually explains why nothing worked.
                raise GemmaClientError(
                    f"Both backends unreachable - {primary_backend}: {last_error}; "
                    f"{fallback_backend}: {fallback_error}"
                ) from fallback_error

        raise last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embeds `texts` via Ollama's real /api/embed endpoint
        (model: settings.gemma_embed_model, default nomic-embed-text).

        Ollama-only, unlike generate(): there's no cross-backend fallback
        or retry here, and this ALWAYS targets ollama_host regardless of
        settings.gemma_backend - the hosted Gemini-style API used
        elsewhere in this project has no embedding endpoint wired up, and
        src/rag.py (the only caller) is explicitly scoped to local Ollama
        deployments, not a general-purpose embedding client."""
        url = f"{self.settings.ollama_host.rstrip('/')}/api/embed"
        payload = {"model": self.settings.gemma_embed_model, "input": texts}

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GemmaClientError(f"Ollama embeddings unreachable at {url}: {exc}") from exc

        data = response.json()
        if "embeddings" not in data:
            raise GemmaClientError(f"Unexpected Ollama embeddings response shape: {data}")
        return data["embeddings"]

    def warm_up(self) -> None:
        """Sends a minimal throwaway prompt to force the model into memory.

        Not called automatically anywhere - call this manually (or from a
        future setup script) before latency-sensitive work to avoid eating
        a cold-start delay on the first real request.
        """
        self.generate(prompt="Reply with the single word: ok")

    def _generate_ollama(
        self, prompt: str, system: Optional[str], timeout: int, format: Optional[dict] = None,
    ) -> str:
        url = f"{self.settings.ollama_host.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.gemma_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format is not None:
            # Real schema-constrained decoding (confirmed directly against
            # this project's own model before relying on it), not a
            # post-hoc parsing hint - Ollama's own generation is constrained
            # to only ever produce JSON matching this shape.
            payload["format"] = format

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GemmaClientError(f"Ollama backend unreachable at {url}: {exc}") from exc

        data = response.json()
        if "response" not in data:
            raise GemmaClientError(f"Unexpected Ollama response shape: {data}")
        return data["response"]

    def _generate_hosted_api(
        self, prompt: str, system: Optional[str], timeout: int, format: Optional[dict] = None,
    ) -> str:
        # format is intentionally accepted-and-ignored, not raised on: the
        # only caller requesting it (pipeline._maneuver_veto_check) only
        # ever runs when the CONFIGURED backend is "ollama" (see
        # make_decide_node), but generate()'s own cross-backend fallback
        # can still land here if Ollama is briefly unreachable - silently
        # degrading to unstructured text (which that caller already knows
        # to handle) beats raising and losing the fallback response
        # entirely.
        if not self.settings.gemma_api_key:
            raise GemmaClientError("GEMMA_API_KEY is required for the 'api' backend")

        # Key goes in a header, not the URL's query string - a failed
        # request's exception text (which includes the request URL) must
        # never end up containing the key, since that text can land in
        # logs, error messages, or (as happened during testing) a terminal.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemma_model_api}:generateContent"
        headers = {"x-goog-api-key": self.settings.gemma_api_key}
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        payload: dict = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GemmaClientError(f"Hosted Gemma API unreachable: {exc}") from exc

        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise GemmaClientError(f"Unexpected hosted API response shape: {data}") from exc
