"""Local OpenAI-compatible proxy: redacts the request, forwards it, restores the reply.

Only non-streaming chat completions with text messages are supported. A request is forwarded only
if it passes an allowlist, otherwise it gets a 400 and is never forwarded:
- top-level fields: ALLOWED_FIELDS, with the types the OpenAI API documents (numbers, integers,
  a string `model`, a string or list of strings `stop`, `stream` false). null is accepted for any
  optional field and forwarded.
- every message has just a role and a content; a content is a string or a list of parts that have
  just a type "text" and a text.
Fields such as `user`, `metadata`, `prediction` or a message's `name` and `tool_calls` can carry
personal data that this module does not look at. `"stream": true` is refused because a stream
cannot be restored here.

The message texts and the `stop` strings are redacted with one set of placeholders. A stop string
with the same full name as a message gets the same placeholder, so it still stops the model; a
part of a name (only the first name) gets a placeholder of its own and never matches.

Limits: `model` is forwarded as sent, a model name is not redacted. An error body from the
upstream is returned as it is, not restored; it can only contain what the upstream received, so it
has placeholders, not personal data.

The proxy never logs a request or a reply. Run uvicorn with the access log off (the `serve`
command does): it would print the request path and query string, which can hold personal data.
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from greek_pii_redactor.names import Names
from greek_pii_redactor.redact import Placeholders, redact, restore
from greek_pii_redactor.rules import Span

# Local models can take a while to answer; this only stops a hung upstream from holding a request.
UPSTREAM_TIMEOUT_SECONDS = 120
ALLOWED_FIELDS = (
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "stream",
)
NUMBER_FIELDS = ("temperature", "top_p", "presence_penalty", "frequency_penalty")
INTEGER_FIELDS = ("max_tokens", "max_completion_tokens", "seed", "n")
ALLOWED_ROLES = ("system", "developer", "user", "assistant")
ALLOWED_MESSAGE_KEYS = ("role", "content")
ALLOWED_PART_KEYS = ("type", "text")


def error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status_code)


def reject_constant(name: str) -> None:
    # json.loads accepts NaN and Infinity by default; they are not JSON and the upstream may choke.
    raise ValueError(f"{name} is not valid JSON")


def redact_request(
    body: dict, detect_names: Callable[[str], list[Span]] | None
) -> Placeholders:
    """Checks the request against the allowlist and redacts it in place. Raises ValueError for
    anything not allowed.

    The texts of one request share one set of placeholders, so a person named in the first
    message and the third is one placeholder.
    """
    placeholders: Placeholders = {}

    def redact_text(text: object) -> str:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        redacted, _ = redact(text, detect_names, placeholders)
        return redacted

    for field, value in body.items():
        if field not in ALLOWED_FIELDS:
            raise ValueError(f"field '{field}' is not supported: it is not redacted")
        if value is None:
            continue
        if field == "stream":
            if value is not False:
                raise ValueError("'stream' must be false: a streamed reply cannot be restored")
        elif field in NUMBER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"field '{field}' must be a number")
        elif field in INTEGER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"field '{field}' must be an integer")
        elif field == "model":
            if not isinstance(value, str):
                raise ValueError("field 'model' must be a string")
        elif field == "stop":
            if isinstance(value, str):
                body["stop"] = redact_text(value)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                body["stop"] = [redact_text(item) for item in value]
            else:
                raise ValueError("field 'stop' must be a string or a list of strings")

    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError("messages must be a list of objects")
    for message in messages:
        for key in message:
            if key not in ALLOWED_MESSAGE_KEYS:
                raise ValueError(f"message key '{key}' is not supported: it is not redacted")
        if message.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"message role must be one of {', '.join(ALLOWED_ROLES)}")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError("only text content parts are supported")
                for key in part:
                    if key not in ALLOWED_PART_KEYS:
                        raise ValueError(
                            f"content part key '{key}' is not supported: it is not redacted"
                        )
                part["text"] = redact_text(part.get("text"))
        else:
            message["content"] = redact_text(content)
    return placeholders


def create_app(
    upstream_url: str,
    detect_names: Callable[[str], list[Span]] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    names: Names | None = None,
) -> FastAPI:
    """`transport` is for tests and measurements: it lets the upstream be an ASGI app in process."""
    upstream_endpoint = f"{upstream_url.rstrip('/')}/chat/completions"

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Made here, not at build time, so it belongs to the event loop the server runs on.
        client = httpx.AsyncClient(transport=transport, timeout=UPSTREAM_TIMEOUT_SECONDS)
        async with client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        try:
            body = json.loads(await request.body(), parse_constant=reject_constant)
        except ValueError:
            return error_response(400, "request body is not valid JSON")
        if not isinstance(body, dict):
            return error_response(400, "request body must be a JSON object")
        # A lone surrogate in any key or value would fail later when the request is encoded.
        try:
            json.dumps(body, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            return error_response(400, "request body is not valid UTF-8 (lone surrogate)")

        # Runs on the event loop, so requests are redacted one at a time: the tagger is not
        # known to be safe to call from several threads.
        try:
            placeholders = redact_request(body, detect_names)
        except ValueError as error:
            return error_response(400, str(error))

        headers = {}
        if "authorization" in request.headers:
            headers["Authorization"] = request.headers["authorization"]
        try:
            upstream = await app.state.client.post(upstream_endpoint, json=body, headers=headers)
        except httpx.HTTPError as error:
            # The exception message is not shown to the client, only its class.
            return error_response(
                502, f"could not get a reply from the upstream ({type(error).__name__})"
            )

        if upstream.status_code >= 400:
            return Response(
                upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
            )
        try:
            reply = json.loads(upstream.content, parse_constant=reject_constant)
        except ValueError:
            return error_response(502, "the upstream reply is not valid JSON")
        choices = reply.get("choices") if isinstance(reply, dict) else None
        if not isinstance(choices, list) or not all(
            isinstance(choice, dict) and isinstance(choice.get("message"), dict)
            for choice in choices
        ):
            return error_response(502, "the upstream reply is not a chat completion")
        for choice in choices:
            message = choice["message"]
            if isinstance(message.get("content"), str):
                message["content"] = restore(message["content"], placeholders, names)
        # Not JSONResponse: it writes UTF-8, which fails on a lone surrogate the upstream may send.
        return Response(
            json.dumps(reply), status_code=upstream.status_code, media_type="application/json"
        )

    return app
