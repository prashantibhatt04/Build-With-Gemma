"""Backend-agnostic client for talking to Gemma, local or hosted."""
from __future__ import annotations

from typing import Optional

import requests

from .config import Settings, settings as default_settings


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

    def generate(self, prompt: str, system: Optional[str] = None, timeout: int = 120) -> str:
        primary_backend = self.settings.gemma_backend
        # Unknown-backend config errors fail fast - not transient, so no
        # retry and no cross-backend fallback attempt for those.
        backend_call = self._resolve_backend_call(primary_backend)

        last_error: GemmaClientError
        for _ in range(2):  # original attempt + one same-backend retry
            try:
                text = backend_call(prompt, system, timeout)
                self.last_backend_used = primary_backend
                return text
            except GemmaClientError as exc:
                last_error = exc

        # Primary backend exhausted its retry. Before falling through to the
        # caller's deterministic fallback, try the other backend once, if
        # it's actually configured (e.g. don't attempt "api" with no key).
        fallback_backend = self._other_backend(primary_backend)
        if fallback_backend is not None and self._backend_configured(fallback_backend):
            fallback_call = self._resolve_backend_call(fallback_backend)
            try:
                text = fallback_call(prompt, system, timeout)
                self.last_backend_used = fallback_backend
                return text
            except GemmaClientError:
                pass  # both backends failed - raise the primary's error below

        raise last_error

    def warm_up(self) -> None:
        """Sends a minimal throwaway prompt to force the model into memory.

        Not called automatically anywhere - call this manually (or from a
        future setup script) before latency-sensitive work to avoid eating
        a cold-start delay on the first real request.
        """
        self.generate(prompt="Reply with the single word: ok")

    def _generate_ollama(self, prompt: str, system: Optional[str], timeout: int) -> str:
        url = f"{self.settings.ollama_host.rstrip('/')}/api/generate"
        payload = {
            "model": self.settings.gemma_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise GemmaClientError(f"Ollama backend unreachable at {url}: {exc}") from exc

        data = response.json()
        if "response" not in data:
            raise GemmaClientError(f"Unexpected Ollama response shape: {data}")
        return data["response"]

    def _generate_hosted_api(self, prompt: str, system: Optional[str], timeout: int) -> str:
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
