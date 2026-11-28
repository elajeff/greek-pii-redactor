"""Sends one chat request to a local Ollama server, for the eval scripts that ask a model."""

import json
import urllib.error
import urllib.request

# Stops a hung server from blocking the run forever.
REQUEST_TIMEOUT_SECONDS = 120


def ask_model(url: str, model: str, seed: int, instruction: str, message: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": message},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": seed},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/api/chat", body, {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw_reply = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode()
        # Ollama answers 404 for an unknown URL too, so the body has to name the model.
        if error.code == 404 and "model" in detail and "not found" in detail:
            raise SystemExit(
                f"Model {model} not found on {url}. Pull it: ollama pull {model}"
            ) from error
        raise SystemExit(
            f"Ollama at {url} returned HTTP {error.code}: {detail}"
        ) from error
    except OSError as error:
        # urlopen wraps a connect timeout in URLError but lets a read timeout through bare.
        timed_out = isinstance(error, TimeoutError) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, TimeoutError)
        )
        if timed_out:
            raise SystemExit(
                f"Ollama at {url} did not answer within {REQUEST_TIMEOUT_SECONDS} s"
            ) from error
        raise SystemExit(f"Cannot reach Ollama at {url}: {error}") from error

    unexpected = f"Ollama at {url} sent an unexpected reply"
    try:
        content = json.loads(raw_reply)["message"]["content"]
    except (ValueError, KeyError, TypeError) as error:
        # Not repr(error): for a body that isn't UTF-8 it holds the whole body.
        raise SystemExit(f"{unexpected} ({type(error).__name__}: {error})") from error
    if not isinstance(content, str):
        raise SystemExit(f"{unexpected} (content is not a string)")
    return content
