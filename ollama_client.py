"""Small, local-only client for Ollama's HTTP API."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"
DEFAULT_TIMEOUT_SECONDS = 120


class OllamaError(RuntimeError):
    """An actionable error communicating with the local Ollama service."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def get_base_url():
    base_url = os.environ.get("SCREENARCH_OLLAMA_URL", DEFAULT_BASE_URL).rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise OllamaError(
            "ScreenArch only sends OCR to a local Ollama endpoint. "
            "Set SCREENARCH_OLLAMA_URL to an http://localhost, http://127.0.0.1, "
            "or http://[::1] Ollama address."
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OllamaError("SCREENARCH_OLLAMA_URL must be a local base URL without credentials or query parameters.")
    return base_url


def _request(method, path, payload=None, timeout=DEFAULT_TIMEOUT_SECONDS):
    url = f"{get_base_url()}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        opener = build_opener(ProxyHandler({}), _RejectRedirects())
        with opener.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        raise OllamaError(
            f"Ollama returned HTTP {exc.code}. Check that the local Ollama service is running."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise OllamaError(
            "Could not reach local Ollama at "
            f"{get_base_url()}. Start Ollama and verify that it is listening on port 11434."
        ) from exc
    try:
        response = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned malformed JSON. Update or restart Ollama and try again.") from exc
    if not isinstance(response, dict):
        raise OllamaError("Ollama returned malformed JSON; expected an object response.")
    return response


def list_models(timeout=10):
    response = _request("GET", "/api/tags", timeout=timeout)
    models = response.get("models")
    if not isinstance(models, list):
        raise OllamaError("Ollama's model list response was missing its models field.")
    names = []
    for model in models:
        if isinstance(model, dict) and isinstance(model.get("name"), str):
            names.append(model["name"])
    return names


def require_model_available(model, timeout=10):
    if not model.strip():
        raise OllamaError("Enter the name of an Ollama model installed locally.")
    model = model.strip()
    if not model.casefold().startswith("gemma"):
        raise OllamaError(
            "ScreenArch classification requires a locally installed Gemma model. "
            "Choose an exact Gemma model name shown by `ollama list`."
        )
    models = list_models(timeout=timeout)
    if model not in models:
        raise OllamaError(
            f"Model '{model}' is not installed in local Ollama. Install a Gemma model yourself "
            "(for example, run `ollama pull {model}` in a terminal), then check again. "
            "ScreenArch never downloads models automatically."
        )
    return models


def generate(model, prompt, system, timeout=DEFAULT_TIMEOUT_SECONDS):
    response = _request(
        "POST",
        "/api/generate",
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    output = response.get("response")
    if not isinstance(output, str):
        raise OllamaError("Ollama's generation response did not contain text.")
    return output
